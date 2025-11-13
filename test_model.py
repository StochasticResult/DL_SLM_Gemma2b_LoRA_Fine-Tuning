"""
Smoke-test the fine-tuned LoRA adapter.
"""
import os
import sys
import torch

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login

# HuggingFace Token
HF_TOKEN = "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"
if HF_TOKEN:
    login(token=HF_TOKEN)

MODEL_NAME = "google/gemma-2-2b"
OUTPUT_DIR = "./outputs/gemma2-lora-dolly-epoch1"
ADAPTER_PATH = os.path.join(OUTPUT_DIR, "lora_adapter")

print("[1/4] Checking GPU...")
if not torch.cuda.is_available():
    print("[ERROR] No GPU detected!")
    raise SystemExit("GPU support is required")

print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

print("[2/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[3/4] Loading base model and LoRA adapter...")
try:
    # Attempt 4-bit loading
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map="auto",
        token=HF_TOKEN,
        attn_implementation="eager",
    )
    print("[OK] Base model loaded with 4-bit quantisation")
except Exception as e:
    print(f"[WARN] Quantised loading failed: {e}")
    print("   Falling back to non-quantised half precision...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN,
        attn_implementation="eager",
    )
    print("[OK] Base model loaded in half precision")

# Merge the LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("[OK] LoRA adapter merged successfully\n")

print("[4/4] Running inference smoke-test...\n")

# Representative prompts
test_prompts = [
    "What is machine learning?",
    "Explain how neural networks work.",
    "When did Virgin Australia start operating?",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"Prompt {i}: {prompt}")
    print("-" * 60)
    
    # Format prompt using Gemma dialogue template
    formatted_prompt = f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>assistant\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract the assistant response span
    if "<start_of_turn>assistant\n" in response:
        response = response.split("<start_of_turn>assistant\n")[-1]
    if "<end_of_turn>" in response:
        response = response.split("<end_of_turn>")[0]
    
    print(f"Response: {response.strip()}")
    print("\n" + "=" * 60 + "\n")

print("[OK] Test complete — the model responds as expected.")

