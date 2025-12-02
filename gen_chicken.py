import torch
from diffusers import StableDiffusionPipeline
import os

# Configuration
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./sd-vangogh-lora"
OUTPUT_FILE = "chicken_vangogh.png"
PROMPT = "a small chicken, oil painting style, van gogh style"

print(f"Initializing Stable Diffusion with base model: {MODEL_ID}")

# Load the pipeline
# Use float16 for GPU memory efficiency
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16,
    safety_checker=None # Disable safety checker to save memory/avoid false positives
)

# Move to GPU
if torch.cuda.is_available():
    pipe.to("cuda")
    print("Pipeline moved to CUDA")
else:
    print("CUDA not available! This will be very slow.")
    pipe.to("cpu")

# Load LoRA weights
print(f"Loading LoRA weights from {LORA_PATH}...")
try:
    pipe.load_lora_weights(LORA_PATH)
    print("LoRA weights loaded successfully.")
except Exception as e:
    print(f"Error loading LoRA: {e}")
    # Fallback or exit? We'll exit as the user specifically asked for the LoRA model
    exit(1)

# Generate image
print(f"Generating image with prompt: '{PROMPT}'...")
image = pipe(PROMPT, num_inference_steps=30).images[0]

# Save image
image.save(OUTPUT_FILE)
print(f"Image saved to {OUTPUT_FILE}")

