import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import os
import sys

# Configuration
MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./sd-vangogh-lora"
OUTPUT_FILE = "chicken_vangogh.png"
PROMPT = "a small chicken, oil painting style, van gogh style"

print(f"Initializing Stable Diffusion with base model: {MODEL_ID}")

try:
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16,
        safety_checker=None
    )
except Exception as e:
    print(f"Error loading base model: {e}")
    sys.exit(1)

# Move to GPU
if torch.cuda.is_available():
    pipe.to("cuda")
    print("Pipeline moved to CUDA")
else:
    print("Warning: Running on CPU (slow)")

# Load LoRA using PEFT
print(f"Loading LoRA weights from {LORA_PATH} via PEFT...")
try:
    # Load adapter onto the UNet
    # Note: pipe.unet is a UNet2DConditionModel
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)
    
    # Merge weights into the base model to speed up inference and ensure compatibility
    # This converts the PeftModel back to the underlying model with weights merged
    if hasattr(pipe.unet, "merge_and_unload"):
        pipe.unet = pipe.unet.merge_and_unload()
        print("LoRA weights merged successfully.")
    else:
        print("Warning: merge_and_unload not found, using PeftModel directly (might be slower).")
        
except Exception as e:
    print(f"Error loading LoRA with PEFT: {e}")
    sys.exit(1)

# Generate image
print(f"Generating image with prompt: '{PROMPT}'...")
try:
    image = pipe(PROMPT, num_inference_steps=30).images[0]
    
    # Save image
    image.save(OUTPUT_FILE)
    print(f"Image saved to {OUTPUT_FILE}")
except Exception as e:
    print(f"Error generating image: {e}")
    sys.exit(1)

