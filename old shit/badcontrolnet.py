from PIL import Image


from cutter import remove_background
from controlnetmaps import FeatureDetector
from cn_cenzor_inpainter import ControlNetCenzorInpainter

image = Image.open("comrade_119.png")
#control_image = Image.open("comrade_119.png")

import cv2
import numpy as np
from PIL import Image

# Делаем карту линий стандартным OpenCV без внешних нейросетей
image_np = np.array(image)
gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(gray, 100, 200)
control_image = Image.fromarray(cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB))

control_image.save("debug_canny.png") # Загляни сюда — там чистый эскиз

remove_background(picture_path="comrade_119.png", output_path="comrade_119_nobg.png")
image_nobg = Image.open("comrade_119_nobg.png")
#fd = FeatureDetector()
#depth_map = fd.get_depth_map(image_nobg, inject_details=True)

cenzor_inpainter = ControlNetCenzorInpainter()
result = cenzor_inpainter.inpaint_spores(
    image=image,
    alpha_print=image_nobg,
    control_image=control_image,
    prompt="concrete texture on skin, skin covered with concrete, concrete texture, concrete wall, concrete background, concrete surface, concrete pattern, concrete material, concrete structure, concrete design, concrete finish, concrete detail, concrete effect, concrete appearance, concrete look",
    negative_prompt="blurry, statue",
    strength=0.7,
    controlnet_scale=0.5,
    guidance_scale=6.5,
    denoise_steps_coef=1.0,
).save("result.png")