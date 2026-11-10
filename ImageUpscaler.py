from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch

# Load model
model_id = "stabilityai/stable-diffusion-x4-upscaler"

pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

# Local image path
image_path = r"E:\API\static\outputs\turboimage\image_1779504896.png"  # Change this to your image path

# Open local image
low_res_img = Image.open(image_path).convert("RGB")

# Resize if needed (recommended for this model)
low_res_img = low_res_img.resize((128, 128))

prompt = "a white cat"

# Upscale image
upscaled_image = pipeline(
    prompt=prompt,
    image=low_res_img
).images[0]

# Save output
output_path = r"E:\API\static\outputs\turboimage\image_1779504896.png"
upscaled_image.save(output_path)

print(f"Saved upscaled image to: {output_path}")