import os
import time
from urllib.parse import urljoin

import torch
from diffusers import ZImagePipeline
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)

os.makedirs("outputs", exist_ok=True)

print("Loading Z-Image Turbo pipeline...")
device = "cuda"
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to(device)
print("Z-Image Turbo pipeline loaded successfully!")


HEIGHT = int(os.getenv("TURBO_IMAGE_HEIGHT", "1024"))
WIDTH = int(os.getenv("TURBO_IMAGE_WIDTH", "1024"))
INFERENCE_STEPS = int(os.getenv("TURBO_IMAGE_STEPS", "9"))
GUIDANCE_SCALE = float(os.getenv("TURBO_IMAGE_GUIDANCE_SCALE", "0.0"))
SEED = int(os.getenv("TURBO_IMAGE_SEED", "42"))
NUMBER_OF_IMAGES = 1

DEFAULT_PROMPT = (
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, "
    "red floral forehead pattern. Elaborate high bun, golden phoenix headdress, "
    "red flowers, beads. Holds round folding fan with lady, trees, bird. Neon "
    "lightning-bolt lamp, bright yellow glow, above extended left palm. Soft-lit "
    "outdoor night background, silhouetted tiered pagoda, blurred colorful distant lights."
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def generate():
    try:
        print("Received Z-Image Turbo request")
        request_data = request.get_json(silent=True) if request.is_json else {}
        prompt_text = (request.form.get("prompt") or request_data.get("prompt") or "").strip()
        if not prompt_text:
            prompt_text = DEFAULT_PROMPT

        timestamp = int(time.time())
        print("Prompt:", prompt_text)
        print("Pipeline starting...")

        start_time = time.time()
        with torch.inference_mode():
            generated_image = pipe(
                prompt=prompt_text,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=torch.Generator(device).manual_seed(SEED),
            ).images[0]

        total_time = time.time() - start_time
        print("image generation finished in", total_time, "seconds")

        output_folder = "static/outputs/turboimage"
        os.makedirs(output_folder, exist_ok=True)
        image_name = f"image_{timestamp}.png"
        image_path = os.path.join(output_folder, image_name)
        generated_image.save(image_path)

        relative_output_path = f"/{output_folder}/{image_name}".replace("\\", "/")
        output_image_url = urljoin(request.host_url, relative_output_path.lstrip("/"))

        return jsonify(
            {
                "message": f"Generated {NUMBER_OF_IMAGES} image(s) successfully!",
                "time_taken": f"{total_time:.2f} seconds",
                "generation_time": f"{total_time:.2f} seconds",
                "background_removal_time": "0.00 seconds",
                "output_image": relative_output_path,
                "output_image_url": output_image_url,
            }
        )

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090)
