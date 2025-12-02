import os
import sys
import time
import traceback

# Force UTF-8 for stdout/stderr
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

print("Starting script...", flush=True)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login

print("Imports done.", flush=True)

# HuggingFace Token
HF_TOKEN = "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"
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("Logged in to Hugging Face.", flush=True)
    except Exception as e:
        print(f"Login failed: {e}", flush=True)

MODEL_NAME = "google/gemma-2-2b"
OUTPUT_DIR = "./outputs/gemma2b-qlora-dolly"
ADAPTER_PATH = OUTPUT_DIR 
SAVE_FILE = "chat_results_qlora.txt"

def log_and_print(f, text):
    print(text, flush=True)
    if f:
        f.write(text + "\n")
        f.flush()

try:
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        log_and_print(f, f"=== Running Gemma-2-2B + QLoRA (4-bit NF4) Test ===\n")
        log_and_print(f, f"Model Base: {MODEL_NAME}")
        log_and_print(f, f"Adapter Path: {ADAPTER_PATH}")
        log_and_print(f, f"Output File: {SAVE_FILE}\n")

        log_and_print(f, "[1/4] Checking GPU...")
        if not torch.cuda.is_available():
            log_and_print(f, "[ERROR] No GPU detected!")
        else:
            log_and_print(f, f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            log_and_print(f, f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

        log_and_print(f, "[2/4] Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, token=HF_TOKEN)
            log_and_print(f, f"Loaded tokenizer from {OUTPUT_DIR}")
        except Exception as e:
            log_and_print(f, f"Could not load tokenizer from {OUTPUT_DIR}: {e}")
            log_and_print(f, f"Falling back to {MODEL_NAME}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log_and_print(f, "[3/4] Loading base model and LoRA adapter...")
        try:
            # QLoRA 4-bit loading
            # Using bfloat16 which is better for RTX 30 series and might avoid NaN issues if training was bf16
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            log_and_print(f, f"Using compute dtype: {compute_dtype}")
            
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
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
            log_and_print(f, "[OK] Base model loaded with 4-bit quantization (NF4)")
        except Exception as e:
            log_and_print(f, f"[ERROR] Quantized loading failed: {e}")
            raise e

        log_and_print(f, f"Loading adapter from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()
        log_and_print(f, "[OK] LoRA adapter merged successfully\n")

        log_and_print(f, "[4/4] Running conversation test...\n")

        test_prompts = [
            "Hello, who are you?",
            "What is machine learning? Explain it simply.",
            "Can you give me a list of 3 healthy snacks?",
            "Tell me a joke.",
            "What is the capital of France?"
        ]

        for i, prompt in enumerate(test_prompts, 1):
            log_and_print(f, f"User: {prompt}")
            log_and_print(f, "-" * 40)
            
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                    )
                
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                # Parse response
                if "<start_of_turn>model\n" in full_response:
                    response_text = full_response.split("<start_of_turn>model\n")[-1]
                else:
                    response_text = full_response

                if "<end_of_turn>" in response_text:
                    response_text = response_text.split("<end_of_turn>")[0]
                if tokenizer.eos_token in response_text:
                    response_text = response_text.split(tokenizer.eos_token)[0]

                log_and_print(f, f"Gemma: {response_text.strip()}")
            
            except RuntimeError as re:
                if "probability tensor contains either `inf`, `nan`" in str(re):
                    log_and_print(f, "[ERROR] Model generated NaN/Inf values.")
                    log_and_print(f, "Cause: The trained adapter likely has corrupted weights (loss was 0.0/NaN during training).")
                else:
                    log_and_print(f, f"[ERROR] Runtime Error during generation: {re}")
            except Exception as e:
                 log_and_print(f, f"[ERROR] Generation failed: {e}")

            log_and_print(f, "\n" + "=" * 60 + "\n")

        log_and_print(f, "[Done] Results saved to " + SAVE_FILE)

except Exception as main_e:
    print(f"CRITICAL ERROR: {main_e}")
    traceback.print_exc()
