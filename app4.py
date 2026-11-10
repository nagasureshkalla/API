import os
import time,json
import base64
from PIL import Image
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import torch
from PromptGenerator import PromptGenerator
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
from PIL import Image
from Temperature import GPUStatus

app3 = Flask(__name__)

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# Load the pipeline once on startup
print("Loading model pipeline...")
device = "cuda"
dtype = torch.bfloat16
pipeline = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype)
pipeline.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
print("Pipeline loaded successfully!")

# Default settings
HEIGHT = 1024
WIDTH = 1536
DIVIDE_BY = 64
NUMBER_OF_IMAGES = 1


@app3.route("/")
def index():
    return render_template("index.html")

@app3.route("/api/upload", methods=["POST"])
def upload():
    try:
        # Get the uploaded image and prompt
        print("Received upload request")
        uploaded_file = request.files.get("image")
        encoded_string =None
        if not uploaded_file:
            encoded_string= None
        else:
            encoded_string = Image.open(uploaded_file.stream).convert("RGB")
        prompt_text = request.form.get("prompt")
        timestamp = int(time.time())
        # image_bytes = uploaded_file.read()
    
            
        # Convert to base64
        
        print("Got the image and prompt:", prompt_text)
        print("pipeline starting...")
        # Generate image(s)
        start_time = time.time()
        images = pipeline(
            image=encoded_string,
            prompt=prompt_text,
            height=457,
            width=441,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=torch.Generator(device=device).manual_seed(0)
        ).images[0]
        end_time = time.time()
        print("pipeline finished in", end_time - start_time, "seconds")
        # Save generated images
        output_folder = f"static/outputs/app4"
        os.makedirs(output_folder, exist_ok=True)
        imageName=f"image_{timestamp}.png"
        image_path = os.path.join(output_folder, imageName)
        images.save(image_path)
        # for i, img in enumerate(images):
        #     imageName= f"image_{i+1}.png"
        #     img_path = os.path.join(output_folder, imageName)
        #     img.save(img_path)
        #     print(f"Saved generated image: {img_path}")
        full_path = f"{output_folder}/{imageName}"
        return jsonify({
            "message": f"Generated {NUMBER_OF_IMAGES} image(s) successfully!",
            "time_taken": f"{end_time - start_time:.2f} seconds",
            "output_image": full_path
        })

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0", port=5000)
    from threading import Thread
    
    Thread(target=lambda: app3.run(host="0.0.0.0", port=8090)).start()
