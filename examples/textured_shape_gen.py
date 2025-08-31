from PIL import Image
import os

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'Hunyuan3D-2')
print("model_path loaded")
print(model_path)
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
print("pipeline_shapegen loaded")

pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)
print("pipeline_texgen loaded")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
image_path = os.path.join(project_root, 'assets', 'demo2.png')
image = Image.open(image_path).convert("RGBA")
if image.mode == 'RGB':
    rembg = BackgroundRemover()
    image = rembg(image)

mesh = pipeline_shapegen(image=image)[0]
mesh = pipeline_texgen(mesh, image=image)
mesh.export('demo.glb')
