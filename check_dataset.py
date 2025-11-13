"""Inspect dataset preprocessing and tokenisation stats."""
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_NAME = "databricks/databricks-dolly-15k"
MODEL_NAME = "google/gemma-2-2b"

# Load dataset
ds = load_dataset(DATASET_NAME)
train_ds = ds["train"]
print(f"Original dataset size: {len(train_ds)}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token="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")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Format helper
def format_dolly(ex):
    instr = str(ex.get("instruction", "")).strip()
    ctx = str(ex.get("context", "")).strip()
    resp = str(ex.get("response", "")).strip()
    prompt = instr if not ctx else f"{instr}\n\nContext: {ctx}"
    if not prompt:
        prompt = "You are a helpful assistant."
    text = (
        f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n"
        f"<start_of_turn>assistant\n{resp}\n<end_of_turn>"
    )
    return [text]

# Inspect post-tokenisation stats
MAX_LEN = 512
valid_count = 0
filtered_count = 0

print("\nChecking tokenisation for the first 100 samples...")
for i in range(min(100, len(train_ds))):
    formatted = format_dolly(train_ds[i])
    tokens = tokenizer(formatted[0], truncation=True, max_length=MAX_LEN, return_length=True)
    if tokens['length'][0] <= MAX_LEN:
        valid_count += 1
    else:
        filtered_count += 1

print("First 100 samples:")
print(f"  Within limit (length <= {MAX_LEN}): {valid_count}")
print(f"  Truncated (length > {MAX_LEN}): {filtered_count}")

# Scan the full dataset
print("\nScanning entire dataset...")
for i in range(len(train_ds)):
    formatted = format_dolly(train_ds[i])
    tokens = tokenizer(formatted[0], truncation=True, max_length=MAX_LEN, return_length=True)
    if tokens['length'][0] <= MAX_LEN:
        valid_count += 1
    else:
        filtered_count += 1
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i+1}/{len(train_ds)} samples...")

print("\nFinal counts:")
print(f"  Valid samples: {valid_count}")
print(f"  Truncated samples: {filtered_count}")
print(f"  Dataset total: {len(train_ds)}")
