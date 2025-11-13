# Gemma-2-2b LoRA Fine-Tuning Project

This repository demonstrates how to fine-tune Google's Gemma-2-2b model with LoRA (Low-Rank Adaptation).

## Prerequisites

- Python 3.8+
- NVIDIA GPU (RTX 3060 or better, 12 GB VRAM recommended)
- CUDA 12.1+
- Windows 10/11

## Quick Start

### 1. Install Dependencies

Run `setup_and_run.bat`, or install manually:

```bash
# Install CUDA-enabled PyTorch
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install the remaining dependencies
pip install -r requirements.txt
pip install tf-keras
```

### 2. Hugging Face Authentication

Gemma models are gated and require authentication:

1. Visit https://huggingface.co/google/gemma-2-2b and accept the license agreement.
2. Create an access token at https://huggingface.co/settings/tokens.
3. Log in from the terminal:
   ```bash
   huggingface-cli login
   ```
   Enter the token generated in step 2.

### 3. Run Training

```bash
python train_local.py
```

## Configuration

Key parameters in `train_local.py`:

- `MODEL_NAME`: base model name (default `"google/gemma-2-2b"`)
- `MAX_LEN`: max sequence length (default 512; increase to 768 or 1024 if memory allows)
- `EPOCHS`: number of epochs (default 1.0)
- `BATCH_SIZE`: per-device batch size (default 1)
- `GRAD_ACC`: gradient accumulation steps (default 8)
- `LR`: learning rate (default 2e-4)
- `RANK`: LoRA rank (default 16)

## Outputs

After training completes, artifacts appear in `./outputs/gemma2-lora-dolly/`:
- `lora_adapter/`: LoRA adapter weights
- Tokenizer files for inference

## Notes

1. **bitsandbytes on Windows**: If 4-bit loading fails, the script automatically falls back to non-quantized loading.
2. **Memory requirements**: RTX 3060 (12 GB) works, but more VRAM is recommended for faster runs.
3. **Training time**: Expect several hours for a full pass on RTX 3060 hardware.

## Troubleshooting

### Issue: `Cannot access gated repo`
**Fix**: Run `huggingface-cli login` (see Quick Start step 2).

### Issue: `bitsandbytes not supported on Windows`
**Fix**: Switch to 8-bit or full precision loading (requires more VRAM).

### Issue: Out of memory
**Fix**:
- Reduce `MAX_LEN` (e.g., 512)
- Keep `BATCH_SIZE` at 1
- Increase `GRAD_ACC` (e.g., 16)
- Consider a smaller base model

