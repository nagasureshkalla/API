import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image
from rembg import remove
from datetime import datetime
import os

class ImageGenerator:
    def __init__(
        self,
        repo_id="black-forest-labs/FLUX.1-Kontext-dev",
        device="cuda:0",
        height=256,
        width=256,
        num_steps=20,
        guidance_scale=2.5,
        use_bfloat16=True,
        use_sequential_cpu_offload=True,
        use_vae_slicing=True,
        use_vae_tiling=True,
        use_channels_last=True,
        use_xformers=True,
        output_dir="outputs"
    ):
        self.repo_id = repo_id
        self.device = device
        self.height = height
        self.width = width
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Torch optimizations
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        
        # Load pipeline
        self.pipe = FluxKontextPipeline.from_pretrained(
            self.repo_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        self.pipe = self.pipe.to(self.device)
        print("Generator Initialized Successfully")
        # Offloading
        # if use_sequential_cpu_offload:
        #     self.pipe.enable_sequential_cpu_offload()
        # else:
        #     self.pipe = self.pipe.to(self.device)

        # VAE optimizations
        try:
            if use_vae_slicing:
                self.pipe.vae.enable_slicing()
            if use_vae_tiling:
                self.pipe.vae.enable_tiling()
        except Exception as e:
            print(f"[VAE OPT WARNING] {e}")

        # Channels last
        # if use_channels_last:
        #     try:
        #         self.pipe.unet.to(memory_format=torch.channels_last)
        #     except Exception:
        #         pass

        # xFormers
        if use_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✓ xFormers enabled")
            except Exception:
                pass

    def generate(self, input_image_path, prompt):
       # print("Started Generating Image From ",input_image_path)
        start_time = datetime.now()

        input_image = load_image(input_image_path)
        #print("Image Loaded")
        torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            #print("Cehcking for Pipeline to Available")
            images = self.pipe(
                image=input_image,
                prompt=prompt,
                guidance_scale=self.guidance_scale
            ).images
        #print("Image Generated")
        image = images[0]

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.png"
        output_path = os.path.join(self.output_dir, filename)
        image.save(output_path)
        #print("Image Saved")
        # Background removal
        # img = Image.open(output_path)
        # img_no_bg = remove(img)
        # img_no_bg.save(output_path)

        duration = datetime.now() - start_time

        return {
            "output_path": output_path,
            "duration": str(duration),
            "gpu_memory_gb": round(
                torch.cuda.max_memory_allocated() / (1024 ** 3), 2
            )
        }
