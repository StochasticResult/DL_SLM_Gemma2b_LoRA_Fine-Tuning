# Training Runs Summary

## Run 1 – 4-bit LoRA (baseline)
- **Script / config**: `train_local.py` with 4-bit quantisation enabled (`USE_4BIT=True`, `OUTPUT_DIR=./outputs/gemma2-lora-dolly-epoch1`)
- **Dataset split**: train 14,860 / eval 151 (1%)
- **Runtime**: ~6,878 s (≈1.9 steps/s)
- **Metrics**: `train_loss ≈ 0.276`, `eval_loss ≈ 1.654`, perplexity ≈ 5.23
- **GPU usage**: Peak ~14.13 GB (PyTorch). NVML telemetry unavailable.
- **Artifacts**: `./outputs/gemma2-lora-dolly-epoch1/lora_adapter/`, training log in `logs/monitor_log.jsonl`, summary in `training_summary.json`

## Run 2 – FP16 (no quantisation)
- **Script / config**: `train_local.py` with 4-bit disabled (`USE_4BIT=False`, `fp16=False`, `OUTPUT_DIR=./outputs/gemma2-lora-dolly-fp16`)
- **Dataset split**: same as Run 1 (train 14,860 / eval 151)
- **Runtime**: ~6,878 s (≈0.54 steps/s)
- **Metrics**: training diverged (`train_loss ≈ 1.72e8`, `eval_loss = NaN`, gradients became NaN around epoch 0.03)
- **GPU usage**: Peak ~9.37 GB (PyTorch), reserved ≈10 GB. NVML still unavailable.
- **Artifacts**: `./outputs/gemma2-lora-dolly-fp16/lora_adapter/`, monitor log `logs/monitor_log.jsonl`, summary `training_summary.json`
- **Notes**: Need gradient scaling / lower LR for stable FP16 training; current run kept purely for record despite NaN results.

