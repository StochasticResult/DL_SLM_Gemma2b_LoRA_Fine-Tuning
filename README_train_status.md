# Training Status Report

## ✅ Current State

**The model has been fine-tuned locally for 2 epochs and is ready for inference.**  
Training, evaluation, and inference scripts have been verified end-to-end. Artifacts are stored in `./outputs/gemma2-lora-dolly-epoch1/`.

### Highlights
- ✅ Successfully loaded `google/gemma-2-2b` with 4-bit quantization
- ✅ Continued LoRA fine-tuning to approximately epoch 2
- ✅ Logged training metrics every 50 steps and performed a final evaluation run
- ✅ Peak GPU memory usage reached **14.13 GB** (PyTorch statistics)
- ⚠️ NVML shared library is unavailable, so real-time GPU utilization percentages were not captured; only memory and loss metrics are recorded

### Artifact Locations
- LoRA adapter: `./outputs/gemma2-lora-dolly-epoch1/lora_adapter/`
- Tokenizer: `./outputs/gemma2-lora-dolly-epoch1/`
- Training summary: `./outputs/gemma2-lora-dolly-epoch1/training_summary.json`
- Step-by-step log (JSONL): `./outputs/gemma2-lora-dolly-epoch1/logs/monitor_log.jsonl`

## 📊 Training Details

- **Epochs**: 2.0 (resumed from checkpoint, ~3714 total steps)
- **Dataset**: `databricks/databricks-dolly-15k`; train split 14,860 samples, eval split 151 samples (1%)
- **Training loss**: decreased into the 1.30–1.56 range, final `train_loss ≈ 0.276`
- **Evaluation**: `eval_loss ≈ 1.654`, perplexity ≈ **5.23**
- **Throughput**: ~1.91 steps/s with gradient accumulation 8; wall time ≈ 32.4 minutes (1,949 seconds)
- **Trainable parameters**: 20.77 M (≈0.79% of the base model)
- **Memory usage**: ~2.37 GB allocated during training, 14.13 GB peak including evaluation/inference

For the full set of statistics, refer to `training_summary.json`, which captures training, evaluation, and memory metrics.

## ⚙️ Inference & Testing

- After training, `python test_model.py` was executed with three sample prompts; all produced coherent answers.
- Inference uses the same 4-bit quantized base model combined with the saved LoRA adapter.
- Terminal logs contain the sample question-answer pairs for reference.

## ⚠️ Notes

1. **GPU utilization**  
   NVML could not be initialized on Windows, so the monitoring callback does not include `gpu_utilization_pct`. Install `nvml.dll` or switch to Linux if utilization percentages are required.
2. **Logging configuration changes**  
   The latest run sets `logging_steps=50` and disables mid-training evaluation (final evaluation only) to keep training uninterrupted.
3. **High peak memory**  
   Evaluation triggers the 14 GB peak. On smaller GPUs, consider reducing `max_seq_length` or enabling gradient checkpointing.

## 🚀 Using the Model

### Option 1: Run the test script
```bash
python test_model.py
```

### Option 2: Load directly in code
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("./outputs/gemma2-lora-dolly-epoch1")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", device_map="auto")
model = PeftModel.from_pretrained(model, "./outputs/gemma2-lora-dolly-epoch1/lora_adapter")

prompt = "<start_of_turn>user\nYour question here\n<end_of_turn>\n<start_of_turn>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
response = tokenizer.decode(outputs[0])
```

## 📈 Next Steps

1. Increase the number of epochs or expand the training subset to further reduce evaluation loss.
2. Migrate `SFTTrainer` settings into an `SFTConfig` to avoid upcoming deprecation warnings.
3. If GPU memory allows, raise `BATCH_SIZE` (and adjust `GRAD_ACC`) to speed up training.
4. Configure GPU monitoring (NVML, Weights & Biases, etc.) for more comprehensive hardware telemetry.

## ✅ Summary

The end-to-end pipeline is validated: loading, resumed training, evaluation, and inference all succeed. The current adapter generates reasonable answers; fine-tune further if higher quality is required.

