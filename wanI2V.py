import os
import uuid
import torch
import numpy as np

from flask import Flask, render_template, request, send_from_directory
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs/wan21_480p"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

print("Loading model...")

image_encoder = CLIPVisionModel.from_pretrained(
    model_id,
    subfolder="image_encoder",
    torch_dtype=torch.float32
)

vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32
)

pipe = WanImageToVideoPipeline.from_pretrained(
    model_id,
    vae=vae,
    image_encoder=image_encoder,
    torch_dtype=torch.bfloat16
)

pipe.to("cuda")

print("Model loaded successfully.")


@app.route("/", methods=["GET", "POST"])
def index():
    video_file = None

    if request.method == "POST":
        image_file = request.files["image"]
        prompt = request.form["prompt"]

        negative_prompt = request.form.get("negative_prompt", "")

        image_path = os.path.join(
            UPLOAD_FOLDER,
            f"{uuid.uuid4()}_{image_file.filename}"
        )

        image_file.save(image_path)

        image = load_image(image_path)

        max_area = 480 * 832
        aspect_ratio = image.height / image.width

        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]

        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

        image = image.resize((width, height))

        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=81,
            guidance_scale=5.0
        ).frames[0]

        video_file = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, video_file)

        export_to_video(output, output_path, fps=16)

    return render_template("wanImageVideo.html", video_file=video_file)


@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)