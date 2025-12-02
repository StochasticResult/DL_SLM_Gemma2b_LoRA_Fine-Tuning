import sys
import time

print("Debug: Script started")
time.sleep(1)

try:
    import torch
    print("Debug: Torch imported")
except ImportError as e:
    print(f"Debug: Torch import failed: {e}")
    sys.exit(1)

# ==========================================================
# Gemma-2-2B QLoRA fine-tuning on Dolly-15k (Fixed Version)
# ==========================================================
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print("Debug: Imports complete")

# --------------------
# Config
# --------------------
USE_QLORA    = True
MODEL_NAME   = "google/gemma-2-2b"
DATASET_NAME = "databricks/databricks-dolly-15k"
OUTPUT_DIR   = "./outputs/gemma2b-qlora-dolly-fixed"
MAX_LEN      = 768
EPOCHS       = 1
BATCH_SIZE   = 1
GRAD_ACC     = 8
LR           = 2e-4
RANK         = 8      # Reduced rank slightly for speed/memory
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
USE_BF16     = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Tokenizer
# --------------------
print("Debug: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

# --------------------
# Quantization (bitsandbytes)
# --------------------
print("Debug: Setting up config...")
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )
else:
    bnb_config = None

# --------------------
# Base model
# --------------------
print("Debug: Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    device_map="auto",
    attn_implementation="eager"
)
model.gradient_checkpointing_enable()
if USE_QLORA:
    prepare_model_for_kbit_training(model)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# --------------------
# PEFT (LoRA)
# --------------------
print("Debug: Applying PEFT...")
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
peft_config = LoraConfig(
    r=RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# --------------------
# Dataset preprocessing
# --------------------
print("Debug: Processing dataset...")
def format_dolly(ex):
    instr = (ex.get("instruction") or "").strip()
    ctx   = (ex.get("context") or "").strip()
    resp  = (ex.get("response") or "").strip()
    prompt = "You are a helpful assistant."
    if instr:
        prompt = instr if not ctx else f"{instr}\n\nContext: {ctx}"
    elif ctx:
        prompt = ctx
    text = (
        f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n"
        f"<start_of_turn>assistant\n{resp}\n<end_of_turn>"
    )
    return {"text": text}

dataset = load_dataset(DATASET_NAME)
train_data = dataset["train"].map(format_dolly, remove_columns=dataset["train"].column_names)

# --------------------
# TrainingArguments
# --------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=5, 
    max_steps=300,  # Quick training: ~300 steps (approx 30-45 mins)
    save_steps=100,
    save_total_limit=2,
    bf16=USE_BF16,
    fp16=(not USE_BF16),
    gradient_checkpointing=True,
    report_to="none",
    group_by_length=True,
)

# --------------------
# Trainer
# --------------------
print("Debug: Initializing Trainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    dataset_text_field="text",
    max_seq_length=MAX_LEN,
    tokenizer=tokenizer,
)

print("Starting training check...")
trainer.train()

print("Saving model...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[SUCCESS] Adapter saved to: {OUTPUT_DIR}")
