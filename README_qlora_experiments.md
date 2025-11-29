## QLoRA / LoRA Experiments on Gemma-2-2B with Dolly-15k Dataset

### Experiment Scripts and Versions
- **Script file**: `qlora.py`
- **Base model**: `google/gemma-2-2b`
- **Dataset**: `databricks/databricks-dolly-15k`
- **Training method**: PEFT + LoRA/QLoRA, using `trl.SFTTrainer`

### Shared Training Configuration (Common to Both Runs)
- **Max sequence length**: `MAX_LEN = 768`
- **Training epochs**: `EPOCHS = 1.5`
- **Batch size**: `BATCH_SIZE = 1`
- **Gradient accumulation**: `GRAD_ACC = 8`
- **Learning rate**: `LR = 2e-4`
- **LoRA rank**: `RANK = 16`
- **LoRA other parameters**:
  - `LORA_ALPHA = 32`
  - `LORA_DROPOUT = 0.05`
- **Precision settings**:
  - `USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()`
  - `bf16 = USE_BF16`, `fp16 = (not USE_BF16)`
- **Target modules (LoRA injection layers)**:
  - `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

---

### Experiment 1: QLoRA Training (USE_QLORA = True)

- **Configuration switch**: `USE_QLORA = True`
- **Weight quantization method**:
  - Using `BitsAndBytesConfig` for 4-bit NF4 quantization:
    - `load_in_4bit = True`
    - `bnb_4bit_quant_type = "nf4"`
    - `bnb_4bit_use_double_quant = True`
    - `bnb_4bit_compute_dtype = torch.bfloat16 if USE_BF16 else torch.float16`
- **Model loading method**:
  - `AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config, torch_dtype=..., device_map="auto")`
  - Enabled `model.gradient_checkpointing_enable()`
  - Called `prepare_model_for_kbit_training(model)` for k-bit training preparation
- **Output directory**:
  - `OUTPUT_DIR = "./outputs/gemma2b-qlora-dolly"`

#### Training Process Highlights
- Typical log entries (key information only):
  - Multiple prints: `{'loss': 0.0, 'grad_norm': nan, 'learning_rate': ... , 'epoch': ...}`
  - Final statistics: `{'train_runtime': 31289.7552, 'train_samples_per_second': 0.72, 'train_steps_per_second': 0.09, 'train_loss': 0.0, 'epoch': 1.5}`
- **Training Time**:
  - Total runtime: **31,289.76 seconds** (~8.69 hours)
  - Training steps: 2,814 steps (1.5 epochs)
  - Steps per second: **0.09 steps/s**
  - Samples per second: **0.72 samples/s**
- **VRAM Memory Usage**:
  - Peak GPU memory: **7.40 GB** (`Max GPU memory (GB): 7.395251274108887`)
  - Memory allocation: Stable throughout training
- From the logs, **training loss remained at 0.0 and grad_norm was consistently NaN**, indicating potential issues with label/loss calculation in the current configuration (e.g., labels not properly set, or loss masked out). Although the Trainer iterated normally, the model did not effectively update.

#### Training Completion and Exception
- After training completed, the script executed:
  - `trainer.model.save_pretrained(OUTPUT_DIR)`
  - `tokenizer.save_pretrained(OUTPUT_DIR)`
- When printing terminal information, the original code used Unicode text with ✅:
  - `print(f"✅ QLoRA adapter + tokenizer saved to: {OUTPUT_DIR}")`
- On Windows default GBK console, this triggered:
  - `UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'...`
- **Impact and conclusion**:
  - This exception occurred after the save operation, so the LoRA adapter and tokenizer were most likely successfully written to `./outputs/gemma2b-qlora-dolly`.
  - The program exited with a non-zero exit code, but training artifacts are essentially complete; the main issue was terminal output encoding incompatibility.

---

### Experiment 2: Standard LoRA Training (USE_QLORA = False)

- **Configuration switch**: `USE_QLORA = False`
- **Weight quantization method**:
  - No 4-bit quantization used, `bnb_config = None`
  - Model loaded directly in half precision:
    - `AutoModelForCausalLM.from_pretrained(..., torch_dtype=..., device_map="auto")`
- **Model settings**:
  - Also enabled `model.gradient_checkpointing_enable()`
  - Still injected LoRA on the same set of `target_modules`
- **Output directory**:
  - `OUTPUT_DIR = "./outputs/gemma2b-lora-dolly"`

#### Training Process Highlights
- Logs show nearly identical learning rate scheduling and epoch progress prints as the QLoRA version:
  - Multiple occurrences: `{'loss': 0.0, 'grad_norm': nan, 'learning_rate': ... , 'epoch': ...}`
  - Final statistics:
    - `{'train_runtime': 21521.0671, 'train_samples_per_second': 1.046, 'train_steps_per_second': 0.131, 'train_loss': 0.0, 'epoch': 1.5}`
- **Training Time**:
  - Total runtime: **21,521.07 seconds** (~5.98 hours)
  - Training steps: 2,814 steps (1.5 epochs)
  - Steps per second: **0.131 steps/s**
  - Samples per second: **1.046 samples/s**
  - **LoRA version trained faster than QLoRA (steps_per_second increased from ~0.09 to 0.13)**
- **VRAM Memory Usage**:
  - Peak GPU memory: **7.71 GB** (`Max GPU memory (GB): 7.706127643585205`)
  - Memory allocation: Stable throughout training
  - Slightly higher than QLoRA version, consistent with the expectation that "no quantization => slightly larger memory footprint".
- Same as QLoRA version, **loss remained at 0.0 and grad_norm was NaN**, further indicating the issue lies in data/label or loss configuration, not the quantization/LoRA switch itself.

#### Training Completion and Saving
- Since console printing was fixed (removed ✅ Unicode character), the ending message printed normally as:
  - `[SUCCESS] LoRA adapter + tokenizer saved to: ./outputs/gemma2b-lora-dolly`
- Program exited with **exit code 0** normally, confirming:
  - LoRA adapter saved to `./outputs/gemma2b-lora-dolly`
  - Tokenizer also saved to the same structure

---

### Comparison and Current Conclusions from Both Runs

- **Switch differences**:
  - Experiment 1: `USE_QLORA = True`, using 4-bit NF4 quantization + QLoRA.
  - Experiment 2: `USE_QLORA = False`, no quantization, standard LoRA only (half-precision weights).
- **Output directories**:
  - QLoRA version: `./outputs/gemma2b-qlora-dolly`
  - LoRA version: `./outputs/gemma2b-lora-dolly`
- **Training Time** (from logs):
  - QLoRA: **31,289.76 seconds** (~8.69 hours), steps/sec ≈ 0.09
  - LoRA: **21,521.07 seconds** (~5.98 hours), steps/sec ≈ 0.13
  - **Speedup**: LoRA is ~1.45x faster than QLoRA
  - Indicates that under current hardware and configuration, **the non-quantized LoRA version is faster**.
- **VRAM Memory Usage**:
  - QLoRA: Peak **7.40 GB**
  - LoRA: Peak **7.71 GB**
  - **Memory difference**: LoRA uses ~4.2% more memory than QLoRA
  - As expected: **QLoRA saves memory but is slightly slower, LoRA uses slightly more memory but is faster**.
- **Loss and gradient status (same for both runs)**:
  - `loss` always `0.0`
  - `grad_norm` always `NaN`
  - Final `train_loss` also `0.0`
  - Indicates that while the SFT process is running, no effective gradient updates are being produced. For future improvements, it is recommended to focus on checking:
    - `SFTTrainer`'s `format_dolly` and label generation method
    - Whether `input_ids` and `labels` in the dataset are correctly constructed
    - Whether the loss mask was accidentally set to all zeros

---

### Encoding Issues and Fixes Applied

- **Problem source**:
  - Windows terminal defaults to GBK encoding, which does not support Unicode characters like ✅.
  - The original code's print statement contained `✅`, causing `UnicodeEncodeError` at the end of the first QLoRA training run.
- **Fix method**:
  - Changed the ending print to ASCII text without special Unicode, for example:
    - `print(f"[SUCCESS] {mode_str} adapter + tokenizer saved to: {OUTPUT_DIR}")`
- **Current status**:
  - The second LoRA run has verified: the script can complete fully in the current environment and print success messages normally.
  - It is recommended that all future log outputs avoid using Emoji/special symbols not supported by the console.

---

### Future Recommendations (If Continuing to Improve Experiments)

- **Data and loss checking**:
  - Before training, manually take a small batch, directly forward pass & manually calculate loss, confirm it is not 0.
  - Print a small sample of `tokenized_train[0]` or `format_dolly` output to ensure text is correctly constructed as prompt + response.
- **More systematic comparison**:
  - After fixing the loss, compare again:
    - QLoRA vs LoRA loss curves at the same number of steps
    - Training time / memory overhead
    - Generation quality on the same evaluation prompts

This file records the **configuration, process, exceptions, and final status** of the current two complete runs, facilitating future experiment reproduction or continued hyperparameter tuning.

---

## Stable Diffusion Text-to-Image: Van Gogh Style LoRA / QLoRA Experiment Records

### Experiment Scripts and Versions
- **Script file**: `sd_vangogh_qlora.py`
- **Base model**: `runwayml/stable-diffusion-v1-5`
- **Dataset**: `alexnasa/vangogh`
- **Training method**: Inject LoRA / QLoRA adapters into UNet, train only a small number of parameters, freeze VAE and text encoder.

### Shared Training Configuration
- **Input resolution**: `RESOLUTION = 512`
- **Training steps**: `MAX_STEPS = 100` (short profiling version)
- **Batch size**: `TRAIN_BATCH = 1`
- **Gradient accumulation**: `GRAD_ACCUM = 4`
- **Learning rate**: `LEARNING_RATE = 1e-4`
- **LoRA configuration**:
  - `r = 16`
  - `lora_alpha = 32`
  - `lora_dropout = 0.05`
  - `target_modules = ["to_k", "to_q", "to_v", "to_out.0"]`
- **Precision and device**:
  - Using `Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")`
  - VAE and `CLIPTextModel` both frozen on GPU with `torch.bfloat16`

---

### Experiment 3: Stable Diffusion QLoRA (USE_QLORA = True)

- **Configuration switch**: `USE_QLORA = True`
- **Weight quantization method**:
  - Using `BitsAndBytesConfig` for 4-bit NF4 quantization on UNet:
    - `load_in_4bit = True`
    - `bnb_4bit_quant_type = "nf4"`
    - `bnb_4bit_compute_dtype = torch.bfloat16`
- **Model loading method**:
  - `UNet2DConditionModel.from_pretrained(..., quantization_config=bnb_config)`
  - Then called `prepare_model_for_kbit_training(unet)` to support k-bit training and gradient checkpointing
- **Output directory**:
  - `OUTPUT_DIR = "./sd-vangogh-qlora"`

#### Key Training Process Logs
- Key console output:
  - `Loading UNet... Mode: QLoRA (4-bit)`
  - `trainable params: 3,188,736 || all params: 862,709,700 || trainable%: 0.3696%`
  - Monitoring information printed every 10 steps:
    - `[Step 10] VRAM: 4.20GB (Peak) | Speed: 0.64 it/s   Loss: 0.0035`
    - ...
    - `[Step 100] VRAM: 4.20GB (Peak) | Speed: 0.58 it/s  Loss: 0.0314`
- **Training Time**:
  - Total steps: 100
  - Average speed: ~0.61 it/s (range: 0.58–0.64 it/s)
  - **Estimated total training time: ~164 seconds (~2.7 minutes)**
  - Note: This is a short profiling run; full training would require more steps
- **VRAM Memory Usage**:
  - Peak GPU memory: **4.20 GB** (consistent throughout training)
  - Memory allocation: Stable, no memory leaks observed
- Observations:
  - **Loss fluctuated in the range 0.00x ~ 0.5**, indicating noise prediction is learning normally;
  - Peak memory approximately **4.20 GB**, relatively lightweight for SD tasks;
  - Step speed approximately **0.58–0.64 it/s**, 100 steps is a relatively short experiment.

#### Training Completion and Saving
- Ending log:
  - `[SUCCESS] Saved SD LoRA adapters to ./sd-vangogh-qlora (USE_QLORA=True)`
- Notes:
  - QLoRA adapter weights successfully saved to `./sd-vangogh-qlora`;
  - This directory can be combined with the original `runwayml/stable-diffusion-v1-5` for Van Gogh style transfer generation.

(Next, `USE_QLORA` was changed to `False` in the same script, and standard LoRA was run again to compare memory and speed performance.)

---

### Experiment 4: Stable Diffusion Standard LoRA (USE_QLORA = False)

- **Configuration switch**: `USE_QLORA = False`
- **Weight quantization method**:
  - No 4-bit quantization used, UNet loaded directly in half precision:
    - `UNet2DConditionModel.from_pretrained(..., torch_dtype=torch.bfloat16)`
- **Other configuration**:
  - All other training hyperparameters (batch, steps, LR, LoRA r, etc.) are identical to Experiment 3 for direct comparison.
- **Output directory**:
  - `OUTPUT_DIR = "./sd-vangogh-lora"`

#### Key Training Process Logs
- Console output:
  - `Loading UNet... Mode: LoRA (16-bit)`
  - `trainable params: 3,188,736 || all params: 862,709,700 || trainable%: 0.3696%`
  - Monitoring every 10 steps:
    - `[Step 10] VRAM: 2.43GB (Peak) | Speed: 0.72 it/s   Loss: 0.0956`
    - ...
    - `[Step 100] VRAM: 2.43GB (Peak) | Speed: 0.65 it/s  Loss: 0.0576`
- **Training Time**:
  - Total steps: 100
  - Average speed: ~0.69 it/s (range: 0.65–0.72 it/s)
  - **Estimated total training time: ~145 seconds (~2.4 minutes)**
  - Note: This is a short profiling run; full training would require more steps
- **VRAM Memory Usage**:
  - Peak GPU memory: **2.43 GB** (consistent throughout training)
  - Memory allocation: Stable, no memory leaks observed
- Observations:
  - Loss also fluctuated in the range 0.0x ~ 0.4+, learning is normal;
  - Peak memory approximately **2.43 GB**;
  - Step speed approximately **0.65–0.72 it/s**, slightly faster than QLoRA version.

#### Training Completion and Saving
- Ending log:
  - `[SUCCESS] Saved SD LoRA adapters to ./sd-vangogh-lora (USE_QLORA=False)`
- Notes:
  - Standard LoRA adapter saved to `./sd-vangogh-lora`, can be combined with base SD1.5 for comparison of generation effects.

---

### Stable Diffusion Image Experiments: QLoRA vs LoRA Comparison Summary

- **Switch differences**:
  - Experiment 3: `USE_QLORA = True`, UNet uses 4-bit NF4 quantization;
  - Experiment 4: `USE_QLORA = False`, UNet loaded directly in bfloat16, no quantization.
- **Memory usage** (peak, from both `Profiler.print_stats`):
  - QLoRA: approximately **4.20 GB**
  - LoRA: approximately **2.43 GB**
  - Under this machine's environment, **standard LoRA actually uses less memory**, because the QLoRA path has additional k-bit training overhead (e.g., extra caching, preparation logic), and both only apply LoRA to the UNet portion.
- **Training Time**:
  - QLoRA: **~164 seconds** (~2.7 minutes) for 100 steps
  - LoRA: **~145 seconds** (~2.4 minutes) for 100 steps
  - **Speedup**: LoRA is ~1.13x faster than QLoRA for the same number of steps
- **Training speed (steps/s)**:
  - QLoRA: approximately **0.58–0.64 it/s** (average ~0.61 it/s)
  - LoRA: approximately **0.65–0.72 it/s** (average ~0.69 it/s)
  - LoRA is overall slightly faster, consistent with the intuition that "simpler operators, no quantization overhead".
- **Loss curve (rough observation)**:
  - Both methods' loss fluctuated around 0.0x ~ 0.5, no obvious convergence difference visible within 100 steps;
  - For such short training (only 100 steps), larger differences would manifest in memory/performance rather than final loss.

Overall conclusions:
- Under this GPU + current implementation, **"QLoRA vs LoRA" for Stable Diffusion is more of a demonstration of performance/memory trade-offs**;
- From a pure resource perspective, LoRA (no quantization) in this experiment is both more memory-efficient and faster;
- If future serious comparison of generation quality is desired, it is necessary to generate the same set of prompts with base vs QLoRA vs LoRA under the same number of steps, then perform subjective or quantitative evaluation.

---

## Summary Table: All Training Results

### Gemma-2-2B Model Training Results

| Configuration | Training Time | VRAM Usage | Training Speed | Steps | Epochs | Output Directory |
|---------------|---------------|------------|----------------|-------|--------|------------------|
| **QLoRA** (4-bit + LoRA) | 31,289.76 sec (~8.69 hours) | 7.40 GB (peak) | 0.09 steps/s | 2,814 | 1.5 | `./outputs/gemma2b-qlora-dolly` |
| **LoRA** (no quantization) | 21,521.07 sec (~5.98 hours) | 7.71 GB (peak) | 0.131 steps/s | 2,814 | 1.5 | `./outputs/gemma2b-lora-dolly` |
| **Speedup** | LoRA is **1.45x faster** | LoRA uses **4.2% more VRAM** | LoRA is **1.46x faster** | Same | Same | - |

### Stable Diffusion Model Training Results

| Configuration | Training Time | VRAM Usage | Training Speed | Steps | Output Directory |
|---------------|---------------|------------|----------------|-------|------------------|
| **QLoRA** (4-bit + LoRA) | ~164 sec (~2.7 min) | 4.20 GB (peak) | 0.58-0.64 it/s (avg ~0.61) | 100 | `./sd-vangogh-qlora` |
| **LoRA** (no quantization) | ~145 sec (~2.4 min) | 2.43 GB (peak) | 0.65-0.72 it/s (avg ~0.69) | 100 | `./sd-vangogh-lora` |
| **Speedup** | LoRA is **1.13x faster** | LoRA uses **42% less VRAM** | LoRA is **1.13x faster** | Same | - |

### Key Observations

1. **Gemma-2-2B**: 
   - QLoRA saves ~4% VRAM but is ~45% slower
   - LoRA uses slightly more VRAM but is significantly faster
   - Both configurations show loss=0.0 issue (needs investigation)

2. **Stable Diffusion**:
   - Counter-intuitively, LoRA uses **less** VRAM than QLoRA (likely due to quantization overhead)
   - LoRA is faster and more memory-efficient for this specific implementation
   - Both show normal loss curves (0.0x ~ 0.5 range)

3. **General Pattern**:
   - Quantization overhead can sometimes outweigh memory savings
   - Actual performance depends on hardware, model size, and implementation details
   - For small models or short training runs, standard LoRA may be preferable
