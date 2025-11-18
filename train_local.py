"""
Local GPU training script for Gemma-2-2b LoRA fine-tuning.
Adapted from the notebook (cell 9) and rewritten to use PEFT utilities directly.
"""
import json
import math
import os
import sys
import time
from typing import Dict, Any, List, Optional

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

try:
    import pynvml  # type: ignore

    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False

# Hugging Face access token (required for gated models)
HF_TOKEN = os.environ.get("HF_TOKEN", "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")
if HF_TOKEN:
    login(token=HF_TOKEN)


class TrainingMonitorCallback(TrainerCallback):
    """Capture GPU and training metrics, persisting them to JSONL and memory."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.records: List[Dict[str, Any]] = []
        self._nvml_handle = None
        self._nvml_available = False
        if torch.cuda.is_available() and HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_available = True
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] Failed to initialise NVML; GPU utilisation will not be recorded: {exc}")

        # Ensure the log directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")  # Truncate any previous log file

    def _get_gpu_metrics(self) -> Dict[str, Any]:
        gpu_info: Dict[str, Any] = {}
        if not torch.cuda.is_available():
            return gpu_info

        try:
            torch.cuda.synchronize()
        except Exception:  # pylint: disable=broad-except
            pass

        try:
            gpu_info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            gpu_info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            gpu_info["gpu_max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            gpu_info["gpu_max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
        except Exception as exc:  # pylint: disable=broad-except
            gpu_info["gpu_memory_error"] = str(exc)

        if self._nvml_available and self._nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                gpu_info["gpu_utilization_pct"] = util.gpu
                gpu_info["gpu_mem_utilization_pct"] = util.memory
                gpu_info["gpu_memory_total_gb"] = mem_info.total / (1024**3)
                gpu_info["gpu_memory_used_gb_nvml"] = mem_info.used / (1024**3)
                gpu_info["gpu_temperature_c"] = pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception as exc:  # pylint: disable=broad-except
                gpu_info["gpu_nvml_error"] = str(exc)
        return gpu_info

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        timestamp = time.time()
        record: Dict[str, Any] = {
            "timestamp": timestamp,
            "epoch": state.epoch,
            "global_step": state.global_step,
        }
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    record[key] = value
        record.update(self._get_gpu_metrics())

        self.records.append(record)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        if self._nvml_available:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # pylint: disable=broad-except
                pass

    def summary(self) -> Dict[str, Any]:
        """Return aggregated statistics for all collected records."""
        if not self.records:
            return {}

        def _collect(key: str) -> List[float]:
            return [float(r[key]) for r in self.records if key in r]

        summary_stats: Dict[str, Any] = {}
        for metric in [
            "loss",
            "learning_rate",
            "grad_norm",
            "eval_loss",
            "gpu_utilization_pct",
            "gpu_mem_utilization_pct",
            "gpu_memory_allocated_gb",
            "gpu_memory_reserved_gb",
        ]:
            values = _collect(metric)
            if not values:
                continue
            summary_stats[f"{metric}_min"] = float(min(values))
            summary_stats[f"{metric}_max"] = float(max(values))
            summary_stats[f"{metric}_avg"] = float(sum(values) / len(values))
        summary_stats["log_samples"] = len(self.records)
        return summary_stats


# === Configuration ===
MODEL_NAME = "google/gemma-2-2b"  # Switch to "google/gemma-2-2b-it" if VRAM is tight
DATASET_NAME = "databricks/databricks-dolly-15k"
OUTPUT_DIR = "./outputs/gemma2-lora-dolly-fp16"  # New FP16 run output directory
MAX_LEN = 512  # Increase to 768 or 1024 if the GPU permits
EPOCHS = 2.0  # Resume for one additional epoch (total ≈2)
BATCH_SIZE = 1
GRAD_ACC = 8
LR = 2e-4
RANK = 16
EVAL_SPLIT = 0.01  # Reserve 1% of the data for validation
EVAL_SEED = 42
MONITOR_LOG_PATH = os.path.join(OUTPUT_DIR, "logs", "monitor_log.jsonl")

# Disable BF16 and prefer FP16 to avoid dtype conversions on unsupported GPUs
USE_BF16 = False  # torch.cuda.is_available() and torch.cuda.is_bf16_supported()
# Enable/disable 4-bit quantisation
USE_4BIT = False

# Validate GPU availability
if not torch.cuda.is_available():
    print("[ERROR] No GPU detected! Please install the CUDA build of PyTorch.")
    print("   Install command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    raise SystemExit("GPU support is required")

print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
print(f"   CUDA version: {torch.version.cuda}")
print(f"   PyTorch version: {torch.__version__}")
print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"   BF16 enabled: {USE_BF16}")
print(f"   NVML telemetry: {'available' if HAS_PYNVML else 'unavailable (missing nvml.dll)'}\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Tokenizer ===
print("[1/7] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, token=HF_TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# === Quantisation / precision configuration ===
print("[2/7] Loading base model...")
if USE_4BIT:
    try:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("   Attempting 4-bit quantisation...")
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_cfg,
            device_map="auto",
            token=HF_TOKEN,
            attn_implementation="eager",  # Force eager attention kernels
        )
        print("[OK] Loaded base model with 4-bit quantisation")
    except Exception as e:  # pylint: disable=broad-except
        print(f"[WARN] 4-bit quantisation failed: {e}")
        print("   Falling back to non-quantised loading (requires more VRAM)...")
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
            device_map="auto",
            token=HF_TOKEN,
            attn_implementation="eager",
        )
        print("[OK] Loaded base model in half precision (no quantisation)")
else:
    print("   Skipping 4-bit quantisation; loading base model in FP16...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
        token=HF_TOKEN,
        attn_implementation="eager",
    )
    print("[OK] Loaded base model directly in FP16")
base.config.pad_token_id = tok.pad_token_id
base.config.use_cache = False

# === LoRA configuration (PEFT) ===
print("[3/7] Configuring LoRA...")
lora_cfg = LoraConfig(
    r=RANK,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()  # Display trainable parameter counts


# === Data preparation ===
def _to_str(x):
    if isinstance(x, list):
        return " ".join(map(str, x))
    if isinstance(x, str):
        return x
    return "" if x is None else str(x)


def format_dolly(ex: Dict[str, Any]) -> List[str]:
    instr = _to_str(ex.get("instruction", "")).strip()
    ctx = _to_str(ex.get("context", "")).strip()
    resp = _to_str(ex.get("response", "")).strip()
    prompt = instr if not ctx else f"{instr}\n\nContext: {ctx}"
    if not prompt:
        prompt = "You are a helpful assistant."
    text = (
        f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n"
        f"<start_of_turn>assistant\n{resp}\n<end_of_turn>"
    )
    return [text]  # TRL expects a list[str]


print("[4/7] Loading and preprocessing dataset...")
ds = load_dataset(DATASET_NAME)
train_ds = ds["train"]
print(f"   Raw dataset size: {len(train_ds)}")

# Preprocess the full dataset into dialogue turns
print("   Formatting samples into chat template...")


def preprocess_dataset(examples):
    texts = []
    for i in range(len(examples["instruction"])):
        formatted = format_dolly(
            {
                "instruction": examples["instruction"][i] if "instruction" in examples else "",
                "context": examples["context"][i] if "context" in examples else "",
                "response": examples["response"][i] if "response" in examples else "",
            }
        )
        texts.append(formatted[0])
    return {"text": texts}


# Apply preprocessing with map
train_ds = train_ds.map(
    preprocess_dataset,
    batched=True,
    remove_columns=train_ds.column_names,
    desc="format dataset",
)
print(f"   Dataset size after formatting: {len(train_ds)}")

# Split validation set
split_dataset = train_ds.train_test_split(test_size=EVAL_SPLIT, seed=EVAL_SEED)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"[INFO] Train size: {len(train_dataset)}, eval size: {len(eval_dataset)} ({EVAL_SPLIT*100:.2f}% of total)")

# Preview a sample
print("\n[INFO] Sample record:")
if len(train_dataset) > 0:
    print(train_dataset[0]["text"][:200].replace("\n", " ⏎ "), "...\n")

# === Training arguments ===
# Calculate expected training steps
expected_steps = len(train_dataset) // (BATCH_SIZE * GRAD_ACC)
print(
    f"[INFO] Expected training steps: {expected_steps} "
    f"(train samples: {len(train_dataset)}, eval samples: {len(eval_dataset)}, "
    f"batch size: {BATCH_SIZE}, grad accumulation: {GRAD_ACC})"
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,  # Train for roughly two epochs
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    max_grad_norm=1.0,
    logging_steps=50,
    logging_strategy="steps",
    evaluation_strategy="no",
    save_strategy="epoch",
    save_steps=500,  # Keep manual save_steps for resume compatibility
    save_total_limit=5,  # Retain several checkpoints for analysis
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.0,
    bf16=USE_BF16,
    fp16=False,
    gradient_checkpointing=False,  # Disable to avoid dtype issues
    report_to="none",
    logging_dir=os.path.join(OUTPUT_DIR, "logs", "tensorboard"),
)

# Prevent half-precision overflow by padding on the right
tok.padding_side = "right"

# Sanity check dataset sizes
print("[DEBUG] Dataset stats after preprocessing...")
print(f"   Train size: {len(train_dataset)}, eval size: {len(eval_dataset)}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=args,
    dataset_text_field="text",  # Explicitly point to the text field
    max_seq_length=MAX_LEN,
    packing=False,
)

monitor_callback = TrainingMonitorCallback(MONITOR_LOG_PATH)
trainer.add_callback(monitor_callback)

# Confirm trainer wiring
print(f"[DEBUG] Trainer ready (train batches: {len(train_dataset)}, eval batches: {len(eval_dataset)})")

# === Train ===
print("[5/7] Starting training...\n")
training_start = time.time()
train_output = trainer.train(resume_from_checkpoint=os.path.isdir(os.path.join(OUTPUT_DIR, "checkpoint-1")))
training_end = time.time()

# Persist training metrics
train_metrics = train_output.metrics or {}
train_runtime = training_end - training_start
train_metrics["train_runtime_seconds_wall"] = train_runtime
trainer.save_metrics("train", train_metrics)
trainer.save_state()

# Final evaluation
print("\n[6/7] Final evaluation...")
eval_metrics = trainer.evaluate()
if "eval_loss" in eval_metrics and eval_metrics["eval_loss"] < float("inf"):
    try:
        eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])
    except OverflowError:
        eval_metrics["eval_perplexity"] = float("inf")
trainer.save_metrics("eval", eval_metrics)

# === Save model ===
print("\n[7/7] Saving model...")
adapter_dir = os.path.join(OUTPUT_DIR, "lora_adapter")
trainer.model.save_pretrained(adapter_dir)
tok.save_pretrained(OUTPUT_DIR)
print(f"[OK] LoRA adapter saved to: {adapter_dir}")

# Summarise GPU monitoring
gpu_summary = monitor_callback.summary()
if torch.cuda.is_available():
    try:
        torch.cuda.synchronize()
    except Exception:  # pylint: disable=broad-except
        pass
    max_memory = torch.cuda.max_memory_allocated() / (1024**3)
else:
    max_memory = 0.0

summary_data: Dict[str, Any] = {
    "train_metrics": train_metrics,
    "eval_metrics": eval_metrics,
    "gpu_monitor_summary": gpu_summary,
    "max_memory_allocated_gb": max_memory,
    "total_training_time_seconds": train_runtime,
    "log_file": MONITOR_LOG_PATH,
    "resume_from_checkpoint": True,
    "epochs_configured": EPOCHS,
    "dataset": {
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "eval_split": EVAL_SPLIT,
    },
}

summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, ensure_ascii=False, indent=2)
print(f"[INFO] Training summary saved to: {summary_path}")
print(f"[INFO] GPU monitor log: {MONITOR_LOG_PATH}")
print(f"[INFO] Peak memory usage (PyTorch): {max_memory:.2f} GB")

print("\n[OK] Training complete!")

