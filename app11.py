import gc
import time
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from diffusers import QwenImageLayeredPipeline

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
MODEL_ID = "Qwen/Qwen-Image-Layered"
IMAGE_PATH = "layer.jpg"

print("=" * 80)
print(f"[{datetime.now()}] Starting application")
print("=" * 80)

# -------------------------------------------------------
# Create output directory
# -------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"static/outputs/app11/Generation{timestamp}")

print(f"[1] Creating output directory...")
output_dir.mkdir(parents=True, exist_ok=True)
print(f"    Output Directory: {output_dir}")

# -------------------------------------------------------
# Clear memory
# -------------------------------------------------------
print(f"[2] Clearing Python and CUDA memory...")
gc.collect()
torch.cuda.empty_cache()

if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    VRAM Total : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"    VRAM Used  : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"    VRAM Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
else:
    print("    CUDA NOT AVAILABLE!")

# -------------------------------------------------------
# Load pipeline
# -------------------------------------------------------
print(f"[3] Loading pipeline...")
start = time.time()

pipeline = QwenImageLayeredPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
print("[3.1] Moving pipeline to CUDA...")
pipeline = pipeline.to("cuda")
print("[3.2] Pipeline moved to CUDA.")
print(f"    Pipeline loaded in {time.time() - start:.2f} seconds")

pipeline.set_progress_bar_config(disable=False)
print("[4] Progress bar enabled.")

# -------------------------------------------------------
# Load image
# -------------------------------------------------------
print(f"[5] Loading image: {IMAGE_PATH}")

image = Image.open(IMAGE_PATH).convert("RGBA")

print(f"    Image Size : {image.size}")
print(f"    Image Mode : {image.mode}")

# -------------------------------------------------------
# Generator
# -------------------------------------------------------
print("[6] Creating CUDA random generator...")

generator = torch.Generator(device="cuda").manual_seed(777)

print("    Seed: 777")

# -------------------------------------------------------
# Inputs
# -------------------------------------------------------
print("[7] Preparing pipeline inputs...")

inputs = {
    "image": image,
    "generator": generator,
    "true_cfg_scale": 3.0,
    "negative_prompt": "",
    "num_inference_steps": 30,
    "num_images_per_prompt": 1,
    "layers": 4,
    "resolution": 640,
    "cfg_normalize": True,
    "use_en_prompt": True,
}

print(inputs)

# -------------------------------------------------------
# Generation
# -------------------------------------------------------
print("=" * 80)
print("[8] STARTING IMAGE GENERATION")
print("=" * 80)

start = time.time()

with torch.inference_mode():
    output = pipeline(**inputs)

elapsed = time.time() - start

print("=" * 80)
print(f"[9] Generation Finished ({elapsed:.2f} seconds)")
print("=" * 80)

# -------------------------------------------------------
# Save
# -------------------------------------------------------
print("[10] Extracting output images...")

output_images = output.images[0]

print(f"    Number of layers: {len(output_images)}")

print("[11] Saving images...")

for i, img in enumerate(output_images):
    output_path = output_dir / f"layer_{i}.png"
    img.save(output_path)
    print(f"    ✓ Saved: {output_path}")

print("=" * 80)
print("[12] DONE")
print(f"Images saved to: {output_dir}")

if torch.cuda.is_available():
    print(f"Final VRAM Used : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Final VRAM Cache: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

print("=" * 80)