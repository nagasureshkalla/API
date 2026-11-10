import os
import time,json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import torch
from PromptGenerator import PromptGenerator
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
from PIL import Image
from Temperature import GPUStatus

app2 = Flask(__name__)
promptgenerator = PromptGenerator()
# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# Load the pipeline once on startup
print("Loading model pipeline...")
pipeline = AutoPipelineForText2Image.from_pretrained(
    "warp-ai/wuerstchen", torch_dtype=torch.float16
).to("cuda")
print("Pipeline loaded successfully!")

# Default settings
HEIGHT = 1024
WIDTH = 1536
DIVIDE_BY = 64
NUMBER_OF_IMAGES = 1

@app2.route("/")
def index():
    return render_template("index.html")

@app2.route("/api/upload", methods=["POST"])
def upload():
    try:
        # Get the uploaded image and prompt
        uploaded_file = request.files.get("image")
        prompt_text = request.form.get("prompt")

        # if not uploaded_file or not prompt_text:
        #     return jsonify({"message": "Missing image or prompt"}), 400

        # # Save uploaded image (optional, can be used as reference)
        timestamp = int(time.time())
        # upload_path = f"static/uploads/{timestamp}_{uploaded_file.filename}"
        # os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        # uploaded_file.save(upload_path)

        # Generate image(s)
        start_time = time.time()
        images = pipeline(
            prompt_text,
            height=HEIGHT,
            width=WIDTH,
            prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
            prior_guidance_scale=4.0,
            num_images_per_prompt=NUMBER_OF_IMAGES,
        ).images
        end_time = time.time()

        # Save generated images
        output_folder = f"static/outputs/app2"
        os.makedirs(output_folder, exist_ok=True)
        for i, img in enumerate(images):
            imageName= f"image_{i+1}.png"
            img_path = os.path.join(output_folder, imageName)
            img.save(img_path)
            print(f"Saved generated image: {img_path}")
        full_path = f"{output_folder}/{imageName}"
        return jsonify({
            "message": f"Generated {NUMBER_OF_IMAGES} image(s) successfully!",
            "time_taken": f"{end_time - start_time:.2f} seconds",
            "output_image": full_path
        })

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app2.route("/autogenerate")
def autogenerate():
    try:
        # 1. Open and load the JSON file
        Number_Of_Times_temperature_Raise=0
        Times_temperature_Raise =[]
        Total_images_Generated =0
        with open("vidogame_assets.json", 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 2. Iterate through the Game Objects by Category
        objects_dict = data.get("video_game_objects", {})
        
        for category, items in objects_dict.items():
            # Formatting the category name for better readability
            #clean_category = category.replace("_", " ").title()
            #print(f"\n[{clean_category}]")
            
            for item in items:
                for style in data["styles"]:
                    try:
                        status=True
                        while status:
                            gpu_status = GPUStatus()
                            gpu_status.update_gpu_stats()
                            temperatures_of_gpus=gpu_status.Get_gpu_temperatures()
                            if any(value > 75 for value in temperatures_of_gpus):
                                print("There is at least one value greater than 75.",temperatures_of_gpus)
                                Number_Of_Times_temperature_Raise=Number_Of_Times_temperature_Raise+1
                                Times_temperature_Raise.append(datetime.now())
                                time.sleep(600)
                                status=True
                            else:
                                status=False
                                if Total_images_Generated>0 and Number_Of_Times_temperature_Raise>0:
                                    time.sleep(150)
                        dynamic_prompt =" change into "+item+" in "+style+" Style"
                        
                        new_prompt = promptgenerator.generate_text(dynamic_prompt)
                        if new_prompt =='' or new_prompt is None:
                            new_prompt=dynamic_prompt
                        print(new_prompt)
                        try:
                            images = pipeline(
                                prompt_text=new_prompt,
                                height=HEIGHT,
                                width=WIDTH,
                                prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
                                prior_guidance_scale=4.0,
                                num_images_per_prompt=NUMBER_OF_IMAGES,
                            ).images
                            timestamp = int(time.time())
                            output_folder = f"static/outputs/{timestamp}"
                            os.makedirs(output_folder, exist_ok=True)
                            for i, img in enumerate(images):
                                imageName= f"image_{i+1}.png"
                                img_path = os.path.join(output_folder, imageName)
                                img.save(img_path)
                                print(f"Saved generated image: {img_path}")
                                Total_images_Generated =Total_images_Generated+1
                        except Exception as e:
                            print("-----------------Temperature History --------------")
                            print("Number of Times High Temperatures Occured : ",Number_Of_Times_temperature_Raise)
                            print("Times of Temperature Raises Occured",Times_temperature_Raise)
                            print(e)
                    except Exception as e:
                        print("-----------------Temperature History --------------")
                        print("Number of Times High Temperatures Occured : ",Number_Of_Times_temperature_Raise)
                        print("Times of Temperature Raises Occured",Times_temperature_Raise)
                        print(e)
                    
                print("-----------------Temperature History --------------")
                print("Number of Times High Temperatures Occured : ",Number_Of_Times_temperature_Raise)
                print("Times of Temperature Raises Occured",Times_temperature_Raise)
                print("total Images Generated ",Total_images_Generated)
    except Exception as er:
        print("-----------------Temperature History --------------")
        print("Number of Times High Temperatures Occured : ",Number_Of_Times_temperature_Raise)
        print("Times of Temperature Raises Occured",Times_temperature_Raise)
        print("Error: Failed to decode JSON. Check your file for syntax errors (commas, brackets).")

if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0", port=5000)
    from threading import Thread
    
    Thread(target=lambda: app2.run(host="0.0.0.0", port=5000)).start()
