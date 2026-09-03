from PIL import Image


from cutter import remove_background
from controlnetmaps import FeatureDetector
from cn_cenzor_inpainter import ControlNetCenzorInpainter

image = Image.open("comrade_119.png")
remove_background(picture_path="comrade_119.png", output_path="comrade_119_nobg.png")
image_nobg = Image.open("comrade_119_nobg.png")
fd = FeatureDetector()
depth_map = fd.get_depth_map(image_nobg, inject_details=True)

cenzor_inpainter = ControlNetCenzorInpainter()
result = cenzor_inpainter.inpaint_spores(
    image=image,
    alpha_print=image_nobg,
    depth_map=depth_map,
    prompt="concrete face",
    negative_prompt="blurry, smooth skin, low quality",
    strength=0.5,
    controlnet_scale=0.75,
    guidance_scale=5.5,
    denoise_steps_coef=1.0,
).save("result.png")