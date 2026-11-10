import os,time
import torch
from flask import Response
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2.1',
    subfolder='hunyuan3d-dit-v2-1',
    use_safetensors=False,
    device='cuda'
)

# paint
#pipeline_paint = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
# mesh = pipeline(
#     image="castle.png",
#     num_inference_steps=30,
#     octree_resolution=380,
#     num_chunks=20000,
#     generator=torch.manual_seed(12345),
#     output_type='trimesh'
# )[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def generate():
    try:
        print("Generating 3D model...")
        # Get image from request
        image = request.files.get("image")
        if not image or image.filename == '':
            return jsonify({"success": False, "message": "No image provided"})
            
        prompt = request.form.get("prompt")
        starttime = time.time()
        
        # Save image temporarily
        image_path = "temp_image.png"
        image.save(image_path)
        
        # Generate 3D model
        mesh = pipeline(
            image=image_path,
            num_inference_steps=300,
            octree_resolution=500,
            num_chunks=20000,
            generator=torch.manual_seed(12345),
            output_type='trimesh'
        )[0]
        # mesh = pipeline_paint(mesh,image)
        # Save mesh
        output_dir = "3DGeneration"
        filename = f"{int(starttime)}.glb"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        mesh.export(output_path)
        endtime = time.time()
        return jsonify({
            "success": True,
            "message": f"Generated 3D model successfully!",
            "time_taken": f"{endtime - starttime:.2f} seconds"
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/getobj', methods=['POST'])
def GetOBJ():
    try:
        print("Generating 3D model...")
        image = request.files.get("image")
        if not image or image.filename == '':
            return jsonify({"success": False, "message": "No image provided"})

        image_path = "temp_image.png"
        image.save(image_path)

        # Generate 3D mesh
        mesh = pipeline(
            image=image_path,
            num_inference_steps=30,
            octree_resolution=380,
            num_chunks=20000,
            generator=torch.manual_seed(12345),
            output_type='trimesh'
        )[0]
        print("Painting 3D model...")
        # mesh = pipeline_paint(mesh,image_path)
        # Export as OBJ bytes
        obj_bytes = mesh.export(file_type='obj').encode('utf-8')
        
        return Response(
            obj_bytes,
            mimetype='text/plain',  # OBJ is text
            headers={
                "Content-Disposition": "attachment; filename=model.obj"
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
    
    
if __name__ == '__main__':
    app.run()




