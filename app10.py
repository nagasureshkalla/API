import os
import time
from urllib.parse import urljoin

from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from diffusers import QwenImageLayeredPipeline

appQwen = Flask(__name__)

# -----------------------------------------------------------------------------
# Create output folder
# -----------------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)

# -----------------------------------------------------------------------------
# Load model once
# -----------------------------------------------------------------------------
print("Loading Qwen Image Layered Pipeline...")

device = "cuda:1"

pipeline = QwenImageLayeredPipeline.from_pretrained(
    "Qwen/Qwen-Image-Layered"
)

pipeline = pipeline.to(device, torch.bfloat16)
pipeline.set_progress_bar_config(disable=None)

print("Pipeline loaded successfully!")

NUMBER_OF_IMAGES = 1

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@appQwen.route("/")
def index():
    return render_template("index.html")


@appQwen.route("/api/upload", methods=["POST"])
def upload():

    try:

        print("Received upload request")

        uploaded_file = request.files.get("image")

        if uploaded_file is None:
            return jsonify({"message": "Image is required"}), 400

        image = Image.open(uploaded_file.stream).convert("RGBA")

        timestamp = int(time.time())

        print("Starting generation...")

        start_time = time.time()

        inputs = {
            "image": image,
            "generator": torch.Generator(device=device).manual_seed(777),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 30,
            "num_images_per_prompt": 1,
            "layers": 2,
            "resolution": 640,
            "cfg_normalize": True,
            "use_en_prompt": True,
        }

        with torch.inference_mode():
            output = pipeline(**inputs)

        output_images = output.images[0]

        end_time = time.time()

        generation_time = end_time - start_time

        print(f"Generation completed in {generation_time:.2f} seconds")

        output_folder = "static/outputs/appQwen"
        os.makedirs(output_folder, exist_ok=True)

        saved_images = []

        for i, img in enumerate(output_images):

            image_name = f"image_{timestamp}_{i}.png"

            image_path = os.path.join(output_folder, image_name)

            img.save(image_path)

            relative_path = f"/{output_folder}/{image_name}".replace("\\", "/")

            saved_images.append(relative_path)

        return jsonify({
            "message": f"Generated {len(saved_images)} layer(s) successfully!",
            "time_taken": f"{generation_time:.2f} seconds",
            "output_images": saved_images,
            "output_image_url": [
                urljoin(request.host_url, path.lstrip("/"))
                for path in saved_images
            ]
        })

    except Exception as e:

        return jsonify({
            "message": f"Error: {str(e)}"
        }), 500


# -----------------------------------------------------------------------------
# Run App
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    from threading import Thread

    Thread(
        target=lambda: appQwen.run(
            host="0.0.0.0",
            port=8090
        )
    ).start()