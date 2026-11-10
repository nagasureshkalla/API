import os
import time
import traceback
from urllib.parse import urljoin

import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, SanaImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads", "sanavideo")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs", "sanavideo")
OUTPUT_URL_PATH = "/static/outputs/sanavideo"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("Loading Sana video pipeline...")
device = "cuda"
pipe = SanaImageToVideoPipeline.from_pretrained(
    "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
)

if os.getenv("SANA_USE_FLOW_SHIFT_SCHEDULER", "0") == "1":
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(
        shift=pipe.scheduler.config.flow_shift
    )

pipe.transformer.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.float32)
pipe.to(device)
print("Sana video pipeline loaded successfully!")


HEIGHT = int(os.getenv("SANA_VIDEO_HEIGHT", "480"))
WIDTH = int(os.getenv("SANA_VIDEO_WIDTH", "832"))
FRAMES = int(os.getenv("SANA_VIDEO_FRAMES", "81"))
FPS = int(os.getenv("SANA_VIDEO_FPS", "16"))
INFERENCE_STEPS = int(os.getenv("SANA_VIDEO_STEPS", "50"))
GUIDANCE_SCALE = float(os.getenv("SANA_VIDEO_GUIDANCE_SCALE", "6"))
MOTION_SCORE = int(os.getenv("SANA_VIDEO_MOTION_SCORE", "30"))
SEED = int(os.getenv("SANA_VIDEO_SEED", "42"))

DEFAULT_PROMPT = (
    "A woman stands against a stunning sunset backdrop, her wavy brown hair "
    "gently blowing in the breeze. She wears a sleeveless, light-colored blouse "
    "with a deep V-neckline, which accentuates her graceful posture. The warm "
    "hues of the setting sun cast a golden glow across her face and hair, "
    "creating a serene and ethereal atmosphere. The background features a "
    "blurred landscape with soft rolling hills and scattered clouds, adding "
    "depth to the scene. The camera remains steady, capturing the tranquil "
    "moment from a medium close-up."
)

DEFAULT_NEGATIVE_PROMPT = (
    "A chaotic sequence with misshapen, deformed limbs, heavy motion blur, "
    "sudden disappearance, jump cuts, jerky movements, bad shot changes, "
    "frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience."
)

DEFAULT_IMAGE_URL = (
    "https://raw.githubusercontent.com/NVlabs/Sana/heads/main/asset/samples/i2v-1.png"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "message": "Sana video Flask app is running",
        }
    )


def _request_value(name, default=""):
    request_data = request.get_json(silent=True) if request.is_json else {}
    return (request.form.get(name) or request_data.get(name) or default).strip()


def _load_request_image():
    image = request.files.get("image")
    if image and image.filename:
        timestamp = int(time.time())
        filename = secure_filename(image.filename)
        image_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{filename}")
        image.save(image_path)
        return load_image(image_path), image_path

    image_url = _request_value("image_url", DEFAULT_IMAGE_URL)
    return load_image(image_url), image_url


def _frame_to_uint8_rgb(frame):
    if hasattr(frame, "convert"):
        return np.array(frame.convert("RGB"))

    frame_array = np.asarray(frame)
    if frame_array.dtype != np.uint8:
        if frame_array.max() <= 1.0:
            frame_array = frame_array * 255
        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)

    if frame_array.ndim == 2:
        frame_array = np.stack([frame_array] * 3, axis=-1)
    if frame_array.shape[-1] == 4:
        frame_array = frame_array[:, :, :3]

    return frame_array


def _export_video(output_frames, output_path, fps):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        saved_video_path = export_to_video(output_frames, output_path, fps=fps)
        saved_video_path = os.path.abspath(saved_video_path)
        if os.path.exists(saved_video_path) and os.path.getsize(saved_video_path) > 0:
            return saved_video_path, "diffusers"
        print("Diffusers exporter did not create a non-empty file:", saved_video_path)
    except Exception:
        print("Diffusers export_to_video failed:")
        traceback.print_exc()

    frames = [_frame_to_uint8_rgb(frame) for frame in output_frames]
    if not frames:
        raise RuntimeError("No frames were produced by the pipeline.")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open MP4 writer for {output_path}")

    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    saved_video_path = os.path.abspath(output_path)
    if not os.path.exists(saved_video_path) or os.path.getsize(saved_video_path) == 0:
        raise RuntimeError(f"OpenCV writer finished, but no MP4 was saved at {saved_video_path}")

    return saved_video_path, "opencv"


@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        input_image, image_source = _load_request_image()
        prompt = _request_value("prompt", DEFAULT_PROMPT)
        negative_prompt = _request_value("negative_prompt", DEFAULT_NEGATIVE_PROMPT)

        if not prompt:
            return jsonify({"message": "Missing prompt"}), 400

        motion_prompt = f" motion score: {MOTION_SCORE}."
        full_prompt = prompt + motion_prompt

        timestamp = int(time.time())
        output_name = f"sana_video_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        print("Generating Sana video:", end=" ", flush=True)
        start_time = time.time()
        with torch.inference_mode():
            output_frames = pipe(
                image=input_image,
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                height=HEIGHT,
                width=WIDTH,
                frames=FRAMES,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=INFERENCE_STEPS,
                generator=torch.Generator(device=device).manual_seed(SEED),
            ).frames[0]

        print("Exporting video to:", output_path)
        saved_video_path, export_backend = _export_video(output_frames, output_path, FPS)

        duration = time.time() - start_time
        print("Done:", saved_video_path)

        relative_output_path = f"{OUTPUT_URL_PATH}/{output_name}"
        output_video_url = urljoin(request.host_url, relative_output_path.lstrip("/"))

        return jsonify(
            {
                "message": "Video generated successfully",
                "output_video": relative_output_path,
                "output_video_url": output_video_url,
                "saved_video_path": saved_video_path,
                "export_backend": export_backend,
                "file_size_bytes": os.path.getsize(saved_video_path),
                "time_taken": f"{duration:.2f} seconds",
                "image_source": image_source,
                "height": HEIGHT,
                "width": WIDTH,
                "frames": FRAMES,
                "fps": FPS,
            }
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8093)
