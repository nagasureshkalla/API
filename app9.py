import os
import time
from urllib.parse import urljoin

import torch
from diffusers import Flux2KleinPipeline
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageFilter
from rembg import new_session, remove


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = os.getenv("APP9_BASE_MODEL", "black-forest-labs/FLUX.2-klein-4B")
LORA_DIR = os.getenv("APP9_LORA_DIR", os.path.join(BASE_DIR, "TrainedCheckpoints"))
LORA_WEIGHT_NAME = os.getenv("APP9_LORA_WEIGHT_NAME", "usoop_flux_lora_v1.safetensors")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs", "app9")
OUTPUT_URL_PATH = "/static/outputs/app9"

app9 = Flask(__name__)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print("Loading app9 model pipeline...")
pipeline = Flux2KleinPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype)

if os.getenv("APP9_CPU_OFFLOAD", "0") == "1" and torch.cuda.is_available():
    pipeline.enable_model_cpu_offload()
else:
    pipeline.to(device)

lora_path = os.path.join(LORA_DIR, LORA_WEIGHT_NAME)
if not os.path.exists(lora_path):
    raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

print("Loading app9 LoRA:", lora_path)
pipeline.load_lora_weights(
    LORA_DIR,
    weight_name=LORA_WEIGHT_NAME,
    adapter_name="custom",
)
pipeline.set_adapters(["custom"], adapter_weights=[float(os.getenv("APP9_LORA_WEIGHT", "1.0"))])

rembg_session = new_session("u2net")
print("app9 pipeline loaded successfully!")


HEIGHT = int(os.getenv("APP9_HEIGHT", "1024"))
WIDTH = int(os.getenv("APP9_WIDTH", "1024"))
GUIDANCE_SCALE = float(os.getenv("APP9_GUIDANCE_SCALE", "3.5"))
INFERENCE_STEPS = int(os.getenv("APP9_INFERENCE_STEPS", "4"))
SEED = int(os.getenv("APP9_SEED", "0"))
MASK_PREVIEW_MAX_SIDE = 768
DEFAULT_PROMPT_SUFFIX = os.getenv(
    "APP9_PROMPT_SUFFIX",
    " Background Grey Screen or Transparent. Centered, fully visible, sharp, clean composition.",
)


def remove_background_preserve_edges(image: Image.Image) -> Image.Image:
    rgba_image = image.convert("RGBA")
    original_width, original_height = rgba_image.size
    preview_image = rgba_image

    longest_side = max(original_width, original_height)
    if longest_side > MASK_PREVIEW_MAX_SIDE:
        scale = MASK_PREVIEW_MAX_SIDE / float(longest_side)
        preview_size = (
            max(1, int(original_width * scale)),
            max(1, int(original_height * scale)),
        )
        preview_image = rgba_image.resize(preview_size, Image.Resampling.LANCZOS)

    transparent_preview = remove(
        preview_image,
        session=rembg_session,
        bgcolor=(0, 0, 0, 0),
        alpha_matting=False,
        post_process_mask=True,
    ).convert("RGBA")

    alpha_mask = transparent_preview.getchannel("A")
    if alpha_mask.size != rgba_image.size:
        alpha_mask = alpha_mask.resize(rgba_image.size, Image.Resampling.LANCZOS)

    alpha_mask = alpha_mask.filter(ImageFilter.MedianFilter(size=3))
    alpha_mask = alpha_mask.filter(ImageFilter.GaussianBlur(radius=0.6))

    red, green, blue, _ = rgba_image.split()
    return Image.merge("RGBA", (red, green, blue, alpha_mask))


def _request_value(name, default=""):
    request_data = request.get_json(silent=True) if request.is_json else {}
    return (request.form.get(name) or request_data.get(name) or default).strip()


@app9.route("/")
def index():
    return render_template("index.cshtml")


@app9.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "message": "app9 LoRA image generator is running",
            "device": device,
            "base_model": BASE_MODEL,
            "lora": lora_path,
        }
    )


@app9.route("/api/images")
def list_images():
    image_items = []
    if os.path.isdir(OUTPUT_FOLDER):
        for file_name in os.listdir(OUTPUT_FOLDER):
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue

            file_path = os.path.join(OUTPUT_FOLDER, file_name)
            if not os.path.isfile(file_path):
                continue

            modified_time = os.path.getmtime(file_path)
            image_items.append(
                {
                    "name": file_name,
                    "url": f"{OUTPUT_URL_PATH}/{file_name}",
                    "modified": modified_time,
                    "modified_label": time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(modified_time),
                    ),
                    "file_size_bytes": os.path.getsize(file_path),
                }
            )

    image_items.sort(key=lambda item: item["modified"], reverse=True)
    return jsonify({"images": image_items})


@app9.route("/api/upload", methods=["POST"])
@app9.route("/api/generate", methods=["POST"])
def upload():
    try:
        uploaded_file = request.files.get("image")
        input_image = None
        if uploaded_file and uploaded_file.filename:
            input_image = Image.open(uploaded_file.stream).convert("RGB")

        prompt_text = _request_value("prompt")
        if not prompt_text:
            return jsonify({"message": "Error: prompt is required"}), 400

        remove_background_value = _request_value(
            "remove_background",
            os.getenv("APP9_REMOVE_BACKGROUND", "0"),
        ).lower()
        should_remove_background = remove_background_value in {"1", "true", "yes", "on"}

        full_prompt = f"{prompt_text.strip()} {DEFAULT_PROMPT_SUFFIX}".strip()
        timestamp = int(time.time())

        print("app9 pipeline starting...")
        start_time = time.time()
        generation_kwargs = {
            "prompt": full_prompt,
            "height": HEIGHT,
            "width": WIDTH,
            "guidance_scale": GUIDANCE_SCALE,
            "num_inference_steps": INFERENCE_STEPS,
            "generator": torch.Generator(device=device).manual_seed(SEED),
        }
        if input_image is not None:
            generation_kwargs["image"] = input_image

        with torch.inference_mode():
            generated_image = pipeline(**generation_kwargs).images[0]

        generation_end_time = time.time()
        if should_remove_background:
            generated_image = remove_background_preserve_edges(generated_image)

        end_time = time.time()
        generation_time = generation_end_time - start_time
        background_removal_time = end_time - generation_end_time
        total_time = end_time - start_time

        image_name = f"image_{timestamp}.png"
        image_path = os.path.join(OUTPUT_FOLDER, image_name)
        generated_image.save(image_path)

        relative_output_path = f"{OUTPUT_URL_PATH}/{image_name}"
        output_image_url = urljoin(request.host_url, relative_output_path.lstrip("/"))

        return jsonify(
            {
                "message": "Generated image successfully!",
                "time_taken": f"{total_time:.2f} seconds",
                "generation_time": f"{generation_time:.2f} seconds",
                "background_removal_time": f"{background_removal_time:.2f} seconds",
                "output_image": relative_output_path,
                "output_image_url": output_image_url,
                "saved_image_path": image_path,
                "file_size_bytes": os.path.getsize(image_path),
                "height": HEIGHT,
                "width": WIDTH,
                "num_inference_steps": INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
            }
        )

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    from threading import Thread
    
    Thread(target=lambda: app9.run(host="0.0.0.0", port=8096)).start()
