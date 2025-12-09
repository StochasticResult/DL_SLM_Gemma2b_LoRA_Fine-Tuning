import os
import sys
import torch
import io
import time

# Force UTF-8 for stdout/stderr
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login

# Helper to print and log
SAVE_FILE = "chat_results_epoch1_fixed.txt"
def log(text):
    print(text, flush=True)
    with open(SAVE_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# Clear file
with open(SAVE_FILE, "w", encoding="utf-8") as f:
    f.write("")

# 1. Setup Token
# Try the token, but if it fails, ignore (maybe cached)
HF_TOKEN = "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"
print(f"Debug: Trying login with token ending in ...{HF_TOKEN[-4:]}")
try:
    login(token=HF_TOKEN)
    log("[Init] Logged in to Hugging Face successfully.")
except Exception as e:
    log(f"[Init] Warning: Login failed ({e}). Attempting to proceed with cached credentials/model.")

# 2. Config
MODEL_NAME = "google/gemma-2-2b"
OUTPUT_DIR = "./outputs/gemma2-lora-dolly-epoch1"
# Try the final adapter first
ADAPTER_PATH = os.path.join(OUTPUT_DIR, "lora_adapter")

log(f"=== Gemma-2-2B + LoRA (Epoch 1) Inference ===")
log(f"Base: {MODEL_NAME}")
log(f"Adapter: {ADAPTER_PATH}")

# 3. Load Tokenizer
log("[Load] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, token=HF_TOKEN)
except:
    log("[Load] Falling back to base tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Load Model (4-bit to match training)
log("[Load] Loading base model in 4-bit (NF4)...")
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        attn_implementation="eager"
    )
    log("[Load] Base model loaded.")
except Exception as e:
    log(f"[Error] Failed to load base model: {e}")
    sys.exit(1)

# 5. Load Adapter
log(f"[Load] Loading adapter from {ADAPTER_PATH}...")
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    log("[Load] Adapter merged.")
except Exception as e:
    log(f"[Error] Failed to load adapter: {e}")
    sys.exit(1)

# 6. Chat Loop
questions = [
    "Hello, who are you?",
    "What is machine learning?",
    "Tell me a fun fact about space."
]

for i, q in enumerate(questions, 1):
    log(f"\n[Q{i}] User: {q}")
    
    # Format
    prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract answer
        if "<start_of_turn>model\n" in full_text:
            answer = full_text.split("<start_of_turn>model\n")[-1]
        else:
            answer = full_text
            
        # Clean up
        answer = answer.split("<end_of_turn>")[0].split(tokenizer.eos_token)[0]
        
        log(f"[Q{i}] Gemma: {answer.strip()}")
        
    except Exception as e:
        log(f"[Error] Generation failed: {e}")

log("\n[Done] Script finished.")


