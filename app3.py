import os
import time
from urllib.parse import urljoin
from flask import Flask, render_template, request, jsonify
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageFilter
from rembg import new_session, remove

app3 = Flask(__name__)

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# Load the pipeline once on startup
print("Loading model pipeline...")
device = "cuda"
dtype = torch.bfloat16
pipeline = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype)
pipeline.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
rembg_session = new_session("u2net")
print("Pipeline loaded successfully!")

# Default settings
HEIGHT = 1024
WIDTH = 1536
DIVIDE_BY = 64
NUMBER_OF_IMAGES = 1
MASK_PREVIEW_MAX_SIDE = 768


def remove_background_preserve_edges(image: Image.Image) -> Image.Image:
    rgba_image = image.convert("RGBA")
    original_width, original_height = rgba_image.size
    preview_image = rgba_image

    # Generate the alpha mask on a smaller preview image to keep
    # background removal close to the original 30s latency budget.
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

    # Light smoothing softens halos without noticeably eating into edges.
    alpha_mask = alpha_mask.filter(ImageFilter.MedianFilter(size=3))
    alpha_mask = alpha_mask.filter(ImageFilter.GaussianBlur(radius=0.6))

    red, green, blue, _ = rgba_image.split()
    return Image.merge("RGBA", (red, green, blue, alpha_mask))


@app3.route("/")
def index():
    return render_template("index.html")

@app3.route("/api/upload", methods=["POST"])
def upload():
    try:
        print("Received upload request")
        request_data = request.get_json(silent=True) if request.is_json else {}
        uploaded_file = request.files.get("image")
        input_image = None

        if uploaded_file:
            input_image = Image.open(uploaded_file.stream).convert("RGB")

        prompt_text = (request.form.get("prompt") or request_data.get("prompt") or "").strip()
        if not prompt_text:
            return jsonify({"message": "Error: prompt is required"}), 400

        timestamp = int(time.time())
        print("Got the image and prompt:", prompt_text+" Transparent or Grey Background ")
        print("pipeline starting...")

        start_time = time.time()
        generated_image = pipeline(
            image=input_image,
            prompt=prompt_text,
            height=720,
            width=720,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=torch.Generator(device=device).manual_seed(0)
        ).images[0]
        generation_end_time = time.time()
        generated_image = remove_background_preserve_edges(generated_image)
        end_time = time.time()
        generation_time = generation_end_time - start_time
        background_removal_time = end_time - generation_end_time
        total_time = end_time - start_time
        print("image generation finished in", generation_time, "seconds")
        print("background removal finished in", background_removal_time, "seconds")
        print("total request finished in", total_time, "seconds")

        output_folder = f"static/outputs/app3"
        os.makedirs(output_folder, exist_ok=True)
        image_name = f"image_{timestamp}.png"
        image_path = os.path.join(output_folder, image_name)
        generated_image.save(image_path)

        relative_output_path = f"/{output_folder}/{image_name}".replace("\\", "/")
        output_image_url = urljoin(request.host_url, relative_output_path.lstrip("/"))

        return jsonify({
            "message": f"Generated {NUMBER_OF_IMAGES} image(s) successfully!",
            "time_taken": f"{total_time:.2f} seconds",
            "generation_time": f"{generation_time:.2f} seconds",
            "background_removal_time": f"{background_removal_time:.2f} seconds",
            "output_image": relative_output_path,
            "output_image_url": output_image_url
        })

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0", port=5000)
    from threading import Thread
    
    Thread(target=lambda: app3.run(host="0.0.0.0", port=8090)).start()
