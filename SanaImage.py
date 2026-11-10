import os
import time
from urllib.parse import urljoin

import torch
from diffusers import SanaSprintPipeline
from flask import Flask, jsonify, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs", "sanaimage")
OUTPUT_URL_PATH = "/static/outputs/sanaimage"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("Loading Sana Sprint image pipeline...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipeline = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)
pipeline.to(device)
print("Sana Sprint image pipeline loaded successfully!")


HEIGHT = int(os.getenv("SANA_IMAGE_HEIGHT", "1024"))
WIDTH = int(os.getenv("SANA_IMAGE_WIDTH", "1024"))
INFERENCE_STEPS = int(os.getenv("SANA_IMAGE_STEPS", "2"))
GUIDANCE_SCALE = float(os.getenv("SANA_IMAGE_GUIDANCE_SCALE", "4.5"))
SEED = int(os.getenv("SANA_IMAGE_SEED", "42"))
DEFAULT_PROMPT = "a tiny astronaut hatching from an egg on the moon"


def _request_value(name, default=""):
    request_data = request.get_json(silent=True) if request.is_json else {}
    return (request.form.get(name) or request_data.get(name) or default).strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "message": "Sana Sprint image Flask app is running",
            "device": device,
        }
    )


@app.route("/api/upload", methods=["POST"])
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        prompt = _request_value("prompt", DEFAULT_PROMPT)
        if not prompt:
            return jsonify({"message": "Missing prompt"}), 400

        timestamp = int(time.time())
        output_name = f"sana_image_{timestamp}.png"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        print("Generating Sana Sprint image:", end=" ", flush=True)
        start_time = time.time()
        with torch.inference_mode():
            image = pipeline(
                prompt=prompt,
                height=HEIGHT,
                width=WIDTH,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=INFERENCE_STEPS,
                generator=torch.Generator(device=device).manual_seed(SEED),
            ).images[0]

        image.save(output_path)
        duration = time.time() - start_time
        print("Done:", output_path)

        relative_output_path = f"{OUTPUT_URL_PATH}/{output_name}"
        output_image_url = urljoin(request.host_url, relative_output_path.lstrip("/"))

        return jsonify(
            {
                "message": "Image generated successfully",
                "output_image": relative_output_path,
                "output_image_url": output_image_url,
                "saved_image_path": output_path,
                "file_size_bytes": os.path.getsize(output_path),
                "time_taken": f"{duration:.2f} seconds",
                "prompt": prompt,
                "height": HEIGHT,
                "width": WIDTH,
                "num_inference_steps": INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8094)
