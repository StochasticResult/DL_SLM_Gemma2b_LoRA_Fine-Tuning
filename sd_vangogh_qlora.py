import time
import torch
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from datasets import load_dataset
from torchvision import transforms
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# CONFIGURATION
USE_QLORA      = False   # 现在改为标准 LoRA（不量化），用于与 QLoRA 对比
MODEL_ID       = "runwayml/stable-diffusion-v1-5"  # Base SD 1.5 model
DATASET_ID     = "alexnasa/vangogh"               # Van Gogh Dataset
OUTPUT_DIR     = "./sd-vangogh-lora" if not USE_QLORA else "./sd-vangogh-qlora"
RESOLUTION     = 512
TRAIN_BATCH    = 1
GRAD_ACCUM     = 4
LEARNING_RATE  = 1e-4
MAX_STEPS      = 100    # 和 notebook 相同（短跑版）
CHECKPOINTING  = True


class Profiler:
    """和 notebook 中一致的简单性能/显存监控工具。"""

    def __init__(self):
        self.start_time = None
        self.step_times = []

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()

    def step_end(self):
        self.step_times.append(time.time())

    def print_stats(self, step: int) -> float:
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_max = torch.cuda.max_memory_allocated() / 1024**3
            mem_res = torch.cuda.memory_reserved() / 1024**3
        else:
            mem_alloc = mem_max = mem_res = 0.0

        if len(self.step_times) > 1:
            avg_time = (self.step_times[-1] - self.step_times[0]) / len(self.step_times)
            speed = 1.0 / avg_time
        else:
            speed = 0.0

        print(f"[Step {step}] VRAM: {mem_max:.2f}GB (Peak) | Speed: {speed:.2f} it/s")
        return mem_max


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("❌ 未检测到 GPU，Stable Diffusion 训练需要 CUDA GPU 才能运行。")

    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM, mixed_precision="bf16")
    profiler = Profiler()

    # 1. Scheduler & Tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")

    # 2. VAE & Text Encoder（冻结）
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16)
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 3. UNet（核心可训练部分）
    print(f"Loading UNet... Mode: {'QLoRA (4-bit)' if USE_QLORA else 'LoRA (16-bit)'}")
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", quantization_config=bnb_config)
        unet = prepare_model_for_kbit_training(unet)
    else:
        unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=torch.bfloat16)

    unet.enable_gradient_checkpointing()

    # 4. 注入 LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # 将冻结模块移动到 GPU
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    # 5. 数据集
    dataset = load_dataset(DATASET_ID, split="train")

    def transform_fn(examples):
        images = [transforms.Resize((RESOLUTION, RESOLUTION))(img).convert("RGB") for img in examples["image"]]
        pixel_values = [transforms.ToTensor()(img) * 2.0 - 1.0 for img in images]
        inputs = tokenizer(
            examples["caption"],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids,
        }

    train_dataset = dataset.with_transform(transform_fn)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH, shuffle=True)

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    # 6. 训练循环
    print("\nStarting Training...")
    profiler.start()

    unet.train()
    global_step = 0

    while global_step < MAX_STEPS:
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=torch.bfloat16, device=accelerator.device)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                profiler.step_end()

                if global_step % 10 == 0:
                    peak_mem = profiler.print_stats(global_step)
                    print(f"   Loss: {loss.item():.4f}")

                if global_step >= MAX_STEPS:
                    break

    # 7. 保存 LoRA adapter
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(OUTPUT_DIR)
        print(f"[SUCCESS] Saved SD LoRA adapters to {OUTPUT_DIR} (USE_QLORA={USE_QLORA})")


if __name__ == "__main__":
    main()


