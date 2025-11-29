# ==========================================================
# Gemma-2-2B QLoRA fine-tuning on Dolly-15k  (TRL ≥ 0.25)
# Works on Python 3.12 + PyTorch 2.4 + Transformers ≥ 4.45
# ==========================================================
import os, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --------------------
# Config
# --------------------
USE_QLORA    = False   # Set to False for standard LoRA (no quantization)
MODEL_NAME   = "google/gemma-2-2b"
DATASET_NAME = "databricks/databricks-dolly-15k"
OUTPUT_DIR   = "./outputs/gemma2b-qlora-dolly" if USE_QLORA else "./outputs/gemma2b-lora-dolly"
MAX_LEN      = 768
EPOCHS       = 1.5
BATCH_SIZE   = 1
GRAD_ACC     = 8
LR           = 2e-4
RANK         = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
USE_BF16     = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Tokenizer
# --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------------------
# Quantization (bitsandbytes) - only if USE_QLORA is True
# --------------------
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    )
    print("Using QLoRA (4-bit quantization)")
else:
    bnb_config = None
    print("Using standard LoRA (no quantization)")

# --------------------
# Base model
# --------------------
if USE_QLORA:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.attn_implementation = "eager"

# --------------------
# PEFT (LoRA)
# --------------------
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

# Tokenize
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )

tokenized_train = train_data.map(tokenize, batched=True, remove_columns=["text"])

# --------------------
# TrainingArguments
# --------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    bf16=USE_BF16,
    fp16=(not USE_BF16),
    gradient_checkpointing=True,
    report_to="none",
)

# --------------------
# Trainer (TRL 0.25+)
# --------------------
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
)

# --------------------
# Train
# --------------------
trainer.train()

if torch.cuda.is_available():
    torch.cuda.synchronize()
    print("Max GPU memory (GB):", torch.cuda.max_memory_allocated() / (1024**3))

# --------------------
# Save Adapter + Tokenizer
# --------------------
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
mode_str = "QLoRA" if USE_QLORA else "LoRA"
print(f"[SUCCESS] {mode_str} adapter + tokenizer saved to: {OUTPUT_DIR}")
