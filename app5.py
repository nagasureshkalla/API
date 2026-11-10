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
                "Content-Disposition": "attachment; filename=model.glb"
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
    
@app.route('/api/getglb', methods=['POST'])
def GetGLB():
    try:
        print("Generating 3D model...")

        image = request.files.get("image")
        if not image or image.filename == '':
            return jsonify({"success": False, "message": "No image provided"})

        image_path = "temp_image.png"
        image.save(image_path)

        # Generate mesh
        mesh = pipeline(
            image=image_path,
            num_inference_steps=30,
            octree_resolution=380,
            num_chunks=20000,
            generator=torch.manual_seed(12345),
            output_type='trimesh'
        )[0]

        print("Exporting GLB...")

        # ✅ Export GLB (binary)
        glb_bytes = mesh.export(file_type='glb')

        return Response(
            glb_bytes,
            mimetype='model/gltf-binary',  # ✅ correct MIME
            headers={
                "Content-Disposition": "attachment; filename=model.glb"
            }
        )

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
       
@app.route('/api/getglbfile', methods=['POST'])
def GetGLBFile():
    try:
        file_path = "3DGeneration/test.glb"

        # Check if file exists
        try:
            with open(file_path, "rb") as f:
                glb_bytes = f.read()
        except FileNotFoundError:
            return jsonify({"success": False, "message": f"File not found: {file_path}"}), 404

        # Return as binary GLB response
        return Response(
            glb_bytes,
            mimetype='model/gltf-binary',
            headers={
                "Content-Disposition": "attachment; filename=model.glb"
            }
        )

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/getobjfile', methods=['POST'])
def GetOBJFile():
    try:
        file_path = "3DGeneration/testobj.obj"

        # Check if file exists
        try:
            with open(file_path, "rb") as f:
                obj_bytes = f.read()
        except FileNotFoundError:
            return jsonify({
                "success": False,
                "message": f"File not found: {file_path}"
            }), 404
        print("")
        return Response(
            obj_bytes,
            mimetype='text/plain',  # OBJ is plain text
            headers={
                "Content-Disposition": "attachment; filename=model.obj"
            }
        )

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    #app.run()
    from threading import Thread
    
    Thread(target=lambda: app.run(host="0.0.0.0", port=8090)).start()




