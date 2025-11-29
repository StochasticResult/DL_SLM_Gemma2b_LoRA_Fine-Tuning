## Gemma-2-2B 在 Dolly-15k 上的 QLoRA / LoRA 实验记录

### 实验脚本与版本
- **脚本文件**: `qlora.py`
- **基础模型**: `google/gemma-2-2b`
- **数据集**: `databricks/databricks-dolly-15k`
- **训练方式**: PEFT + LoRA/QLoRA，使用 `trl.SFTTrainer`

### 公共训练配置（两次运行共享）
- **最大长度**: `MAX_LEN = 768`
- **训练轮数**: `EPOCHS = 1.5`
- **batch 大小**: `BATCH_SIZE = 1`
- **梯度累积**: `GRAD_ACC = 8`
- **学习率**: `LR = 2e-4`
- **LoRA rank**: `RANK = 16`
- **LoRA 其它参数**:
  - `LORA_ALPHA = 32`
  - `LORA_DROPOUT = 0.05`
- **精度设置**:
  - `USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()`
  - `bf16 = USE_BF16`, `fp16 = (not USE_BF16)`
- **目标模块（LoRA 注入层）**:
  - `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

---

### 实验一：QLoRA 训练（USE_QLORA = True）

- **配置开关**: `USE_QLORA = True`
- **权重量化方式**:
  - 使用 `BitsAndBytesConfig` 做 4-bit NF4 量化：
    - `load_in_4bit = True`
    - `bnb_4bit_quant_type = "nf4"`
    - `bnb_4bit_use_double_quant = True`
    - `bnb_4bit_compute_dtype = torch.bfloat16 if USE_BF16 else torch.float16`
- **模型加载方式**:
  - `AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config, torch_dtype=..., device_map="auto")`
  - 开启 `model.gradient_checkpointing_enable()`
  - 调用 `prepare_model_for_kbit_training(model)` 做 k-bit 训练准备
- **输出目录**:
  - `OUTPUT_DIR = "./outputs/gemma2b-qlora-dolly"`

#### 训练过程要点
- 日志中可见典型条目（只保留关键信息）：
  - 多次打印：`{'loss': 0.0, 'grad_norm': nan, 'learning_rate': ... , 'epoch': ...}`
  - 最终统计：`{'train_runtime': 31289.7552, 'train_samples_per_second': 0.72, 'train_steps_per_second': 0.09, 'train_loss': 0.0, 'epoch': 1.5}`
  - 最大显存占用：`Max GPU memory (GB): 7.395251274108887`
- 从日志看，**训练 loss 一直为 0.0，grad_norm 始终为 NaN**，说明当前配置下的 label / loss 计算可能存在问题（例如没有正确设置 labels，或者 loss 被屏蔽），虽然 Trainer 正常迭代，但模型实际上没有有效更新。

#### 训练结束与异常情况
- 训练完成后，脚本执行了：
  - `trainer.model.save_pretrained(OUTPUT_DIR)`
  - `tokenizer.save_pretrained(OUTPUT_DIR)`
- 随后在打印终端信息时，由于原始代码中使用了带 ✅ 的 Unicode 文本：
  - `print(f"✅ QLoRA adapter + tokenizer saved to: {OUTPUT_DIR}")`
- 在 Windows 默认 GBK 控制台下，引发：
  - `UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'...`
- **影响与结论**:
  - 该异常发生在保存操作之后的打印阶段，因此 LoRA adapter 和 tokenizer 很大概率已经成功写入 `./outputs/gemma2b-qlora-dolly`。
  - 程序以非零退出码结束，但训练产物基本齐全，主要问题是终端输出编码不兼容。

---

### 实验二：标准 LoRA 训练（USE_QLORA = False）

- **配置开关**: `USE_QLORA = False`
- **权重量化方式**:
  - 不再使用 4-bit 量化，`bnb_config = None`
  - 模型以半精度直接加载：
    - `AutoModelForCausalLM.from_pretrained(..., torch_dtype=..., device_map="auto")`
- **模型设置**:
  - 同样启用了 `model.gradient_checkpointing_enable()`
  - 仍然在同一组 `target_modules` 上注入 LoRA
- **输出目录**:
  - `OUTPUT_DIR = "./outputs/gemma2b-lora-dolly"`

#### 训练过程要点
- 日志中可见与 QLoRA 版本几乎同样的学习率调度与 epoch 进度打印：
  - 多次出现：`{'loss': 0.0, 'grad_norm': nan, 'learning_rate': ... , 'epoch': ...}`
  - 最终统计：
    - `{'train_runtime': 21521.0671, 'train_samples_per_second': 1.046, 'train_steps_per_second': 0.131, 'train_loss': 0.0, 'epoch': 1.5}`
    - 可见 **LoRA 版本训练速度快于 QLoRA（steps_per_second 从约 0.09 提升到 0.13）**
  - 最大显存占用：
    - `Max GPU memory (GB): 7.706127643585205`
    - 显存略高于 QLoRA 版本，符合“无量化 => 显存占用略大”的预期。
- 与 QLoRA 版本一样，**loss 始终为 0.0，grad_norm 为 NaN**，这进一步说明问题出在数据 / label 或 loss 配置，而不是量化 / LoRA 开关本身。

#### 训练结束与保存
- 由于已经修复了控制台打印（去掉了 ✅ Unicode 字符），结尾信息正常打印为：
  - `[SUCCESS] LoRA adapter + tokenizer saved to: ./outputs/gemma2b-lora-dolly`
- 程序以 **退出码 0** 正常结束，确认：
  - LoRA adapter 已保存到 `./outputs/gemma2b-lora-dolly`
  - tokenizer 也保存到了同一结构中

---

### 两次运行的对比与当前结论

- **开关差异**:
  - 实验一：`USE_QLORA = True`，使用 4-bit NF4 量化 + QLoRA。
  - 实验二：`USE_QLORA = False`，不量化，只做标准 LoRA（半精度权重）。
- **输出目录**:
  - QLoRA 版本：`./outputs/gemma2b-qlora-dolly`
  - LoRA 版本：`./outputs/gemma2b-lora-dolly`
- **训练时间**（来自日志）:
  - QLoRA：约 31,290 秒（约 8.7 小时），steps/sec ≈ 0.09
  - LoRA：约 21,521 秒（约 6 小时），steps/sec ≈ 0.13
  - 说明在当前硬件与配置下，**不量化的 LoRA 版本速度更快**。
- **显存占用**:
  - QLoRA：峰值约 7.40 GB
  - LoRA：峰值约 7.71 GB
  - 符合预期：**QLoRA 省显存但稍慢，LoRA 多占一点显存但更快**。
- **loss 与梯度情况（两次均相同）**:
  - `loss` 始终为 `0.0`
  - `grad_norm` 始终为 `NaN`
  - 最终 `train_loss` 也为 `0.0`
  - 说明目前 SFT 过程虽然在跑，但没有产生有效梯度更新，后续如果需要真正提升效果，建议重点检查：
    - `SFTTrainer` 的 `format_dolly` 与 label 生成方式
    - 数据集中 `input_ids`、`labels` 是否被正确构造
    - 是否没有意外地将 loss mask 全部置为 0

---

### 编码问题与已做的修复

- **问题来源**:
  - Windows 终端默认使用 GBK 编码，不支持 ✅ 等 Unicode 字符。
  - 原始代码中的打印语句含有 `✅`，导致第一轮 QLoRA 训练结束时报 `UnicodeEncodeError`。
- **修复方式**:
  - 将结尾打印改为不含特殊 Unicode 的 ASCII 文本，例如：
    - `print(f"[SUCCESS] {mode_str} adapter + tokenizer saved to: {OUTPUT_DIR}")`
- **当前状态**:
  - 第二次 LoRA 运行已经验证：脚本可在当前环境下完整跑完并正常打印成功信息。
  - 建议后续所有日志输出尽量避免使用控制台不支持的 Emoji / 特殊符号。

---

### 后续建议（如果要继续改进实验）

- **数据与 loss 检查**:
  - 在训练前，手动取一小 batch，直接前向 & 手算 loss，确认不为 0。
  - 打印一小段 `tokenized_train[0]` 或 `format_dolly` 输出，确保文本被正确构造成 prompt + response。
- **更系统的对比**:
  - 在修复 loss 之后，可以再次对比：
    - QLoRA vs LoRA 在同样 steps 下的 loss 曲线
    - 训练时间 / 显存开销
    - 在相同 evaluation prompt 上的生成质量

本文件记录了当前两次完整运行的**配置、过程、异常与最终状态**，便于之后复现实验或继续调参。  

---

## Stable Diffusion 文本生成图像：Van Gogh 风格 LoRA / QLoRA 实验记录

### 实验脚本与版本
- **脚本文件**: `sd_vangogh_qlora.py`
- **基础模型**: `runwayml/stable-diffusion-v1-5`
- **数据集**: `alexnasa/vangogh`
- **训练方式**: 对 UNet 注入 LoRA / QLoRA 适配器，仅训练少量参数，VAE 与文本编码器冻结。

### 公共训练配置
- **输入分辨率**: `RESOLUTION = 512`
- **训练步数**: `MAX_STEPS = 100`（短跑 profiling 版本）
- **batch 大小**: `TRAIN_BATCH = 1`
- **梯度累积**: `GRAD_ACCUM = 4`
- **学习率**: `LEARNING_RATE = 1e-4`
- **LoRA 配置**:
  - `r = 16`
  - `lora_alpha = 32`
  - `lora_dropout = 0.05`
  - `target_modules = ["to_k", "to_q", "to_v", "to_out.0"]`
- **精度与设备**:
  - 使用 `Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")`
  - VAE 与 `CLIPTextModel` 均以 `torch.bfloat16` 冻结在 GPU 上

---

### 实验三：Stable Diffusion QLoRA（USE_QLORA = True）

- **配置开关**: `USE_QLORA = True`
- **权重量化方式**:
  - 使用 `BitsAndBytesConfig` 对 UNet 做 4-bit NF4 量化：
    - `load_in_4bit = True`
    - `bnb_4bit_quant_type = "nf4"`
    - `bnb_4bit_compute_dtype = torch.bfloat16`
- **模型加载方式**:
  - `UNet2DConditionModel.from_pretrained(..., quantization_config=bnb_config)`
  - 之后调用 `prepare_model_for_kbit_training(unet)` 以支持 k-bit 训练和梯度检查点
- **输出目录**:
  - `OUTPUT_DIR = "./sd-vangogh-qlora"`

#### 训练过程关键日志
- 控制台关键输出：
  - `Loading UNet... Mode: QLoRA (4-bit)`
  - `trainable params: 3,188,736 || all params: 862,709,700 || trainable%: 0.3696%`
  - 每 10 步打印一次监控信息：
    - `[Step 10] VRAM: 4.20GB (Peak) | Speed: 0.64 it/s   Loss: 0.0035`
    - ...
    - `[Step 100] VRAM: 4.20GB (Peak) | Speed: 0.58 it/s  Loss: 0.0314`
- 观察：
  - **Loss 在 0.00x ~ 0.5 区间波动**，说明噪声预测在正常学习；
  - 峰值显存约 **4.20 GB**，对 SD 任务来说相对轻量；
  - 步速约 **0.58–0.64 it/s**，100 步属于较短实验。

#### 训练结束与保存
- 结尾日志：
  - `[SUCCESS] Saved SD LoRA adapters to ./sd-vangogh-qlora (USE_QLORA=True)`
- 说明：
  - QLoRA 适配器权重已成功保存到 `./sd-vangogh-qlora`；
  - 该目录可与原始 `runwayml/stable-diffusion-v1-5` 组合，用于 Van Gogh 风格迁移生成。

（接下来会在同一脚本中将 `USE_QLORA` 改为 `False`，再跑一遍标准 LoRA 以对比显存与速度表现。） 

---

### 实验四：Stable Diffusion 标准 LoRA（USE_QLORA = False）

- **配置开关**: `USE_QLORA = False`
- **权重量化方式**:
  - 不使用 4-bit 量化，直接以半精度加载 UNet：
    - `UNet2DConditionModel.from_pretrained(..., torch_dtype=torch.bfloat16)`
- **其它配置**:
  - 其余训练超参数（batch、steps、LR、LoRA r 等）与实验三完全一致，方便做一一对比。
- **输出目录**:
  - `OUTPUT_DIR = "./sd-vangogh-lora"`

#### 训练过程关键日志
- 控制台输出：
  - `Loading UNet... Mode: LoRA (16-bit)`
  - `trainable params: 3,188,736 || all params: 862,709,700 || trainable%: 0.3696%`
  - 每 10 步监控：
    - `[Step 10] VRAM: 2.43GB (Peak) | Speed: 0.72 it/s   Loss: 0.0956`
    - ...
    - `[Step 100] VRAM: 2.43GB (Peak) | Speed: 0.65 it/s  Loss: 0.0576`
- 观察：
  - Loss 同样在 0.0x ~ 0.4+ 区间波动，学习正常；
  - 峰值显存约 **2.43 GB**；
  - 步速约 **0.65–0.72 it/s**，略快于 QLoRA 版本。

#### 训练结束与保存
- 结尾日志：
  - `[SUCCESS] Saved SD LoRA adapters to ./sd-vangogh-lora (USE_QLORA=False)`
- 说明：
  - 标准 LoRA 适配器已保存到 `./sd-vangogh-lora`，可与 base SD1.5 结合用于对比生成效果。

---

### Stable Diffusion 图像实验：QLoRA vs LoRA 对比小结

- **开关差异**:
  - 实验三：`USE_QLORA = True`，UNet 使用 4-bit NF4 量化；
  - 实验四：`USE_QLORA = False`，UNet 直接以 bfloat16 加载，无量化。
- **显存占用**（峰值，来自两次 `Profiler.print_stats`）:
  - QLoRA：约 **4.20 GB**
  - LoRA：约 **2.43 GB**
  - 在本机环境下，**反而是标准 LoRA 更省显存**，原因是 QLoRA 路径中多了一些 k-bit 训练额外开销（如额外缓存、准备逻辑），并且两边都只对 UNet 部分做 LoRA。
- **训练速度（steps/s）**:
  - QLoRA：约 **0.58–0.64 it/s**
  - LoRA：约 **0.65–0.72 it/s**
  - LoRA 整体略快一些，与“算子更简单，无量化开销”的直觉一致。
- **损失曲线（粗观察）**:
  - 两种方式的 loss 都在 0.0x ~ 0.5 左右波动，100 步内看不出明显收敛差异；
  - 对于这么短的训练（仅 100 步），更大的差别会体现在显存 / 性能而非最终 loss。

整体结论：  
- 在当前这台 GPU + 当前实现方式下，**Stable Diffusion 的“QLoRA vs LoRA”更多是性能/显存 trade-off 的演示**；  
- 从纯资源角度看，LoRA（无量化）在本实验中既更省显存又更快一步；  
- 如果后续想严肃对比生成质量，需要在相同步数的前提下，用 base vs QLoRA vs LoRA 生成同一组 prompt，再做主观或定量评测。 
