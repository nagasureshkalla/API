
from flask import Flask, render_template, jsonify,request
import os
import json
import torch
import time
from datetime import datetime
from ImageGeneration import ImageGenerator
from PromptGenerator import PromptGenerator
from Temperature import GPUStatus

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/app"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# print("Torch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA device count:", torch.cuda.device_count())


# 🔥 Load model ONCE (important)
generator = ImageGenerator()
promptgenerator = PromptGenerator()

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/autogenerate")
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
                            
                            generator.generate(
                                input_image_path=os.path.join(UPLOAD_FOLDER, "nika.png"),
                                prompt=new_prompt
                            )
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

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "Flask app is running"
    })

@app.route("/api/upload", methods=["POST"])
def upload():
    image = request.files.get("image")
    prompt = request.form.get("prompt")

    if not image or not prompt:
        return jsonify({"message": "Missing image or prompt"}), 400

    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    # Here you can run ML / image processing logic
    # print("Prompt:", prompt)
    # print("Saved Image:", image_path)
    print("Generating Iamge :",end=" ", flush=True)
    new_prompt = promptgenerator.generate_text(prompt)
    if new_prompt =='' or new_prompt is None:
        new_prompt=prompt
    try:
       
        result = generator.generate(
            input_image_path=image_path,
            prompt=new_prompt
        )
        print(" Done")
        return jsonify({
            "message": "Image generated successfully",
            "output_image": result["output_path"],
            "time_taken": result["duration"],
            "gpu_memory_gb": result["gpu_memory_gb"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0", port=6000)
    from threading import Thread
    
    Thread(target=lambda: app.run(host="0.0.0.0", port=8080)).start()
