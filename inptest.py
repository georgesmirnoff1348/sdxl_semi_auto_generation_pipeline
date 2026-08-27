from PIL import Image, ImageDraw
from composer import Composer
from inpainter import Inpainter
from cutter import remove_background

"""
def inpaint(
        self,
        composition: CompositionResult = None,
        background: Image.Image = None,
        figure: Image.Image = None,
        prompt: str = "realistic photo of a man, seamless integration, high quality, realistic lighting, soft shadows",
        negative_prompt: str = "blurry, low quality, sharp edges, artifacts, ugly distortion",
        strength: float = 0.65,
        denoise_steps_coef: float = 1.0,
        guidance_scale: float = 7.5,
        seed: int = None
    ) -> Image.Image
"""

fig0 = remove_background("babka.jpg", "babka.png")
fig = Image.open("babka.png")
bg = Image.open("cyberpunk.jpg")

comp = Composer(interactive=True, verbose=False)

collage_and_mask = comp.compose(
    background=bg,
    figure=fig,
    position=(0, 0),
    mode="background_primary",
    scale=1.8)

#collage_and_mask.collage.show()
collage_and_mask.collage.save("collage.png")
#collage_and_mask.mask.show()

inpainter = Inpainter()

inpainted_image = inpainter.inpaint(
    composition=collage_and_mask,
    prompt="realistic granny in cyberpunk environment, neon lighting, cyberpunk high quality, realistic lighting, soft shadows, " \
    "seamless integration, photorealistic, cinematic lighting, ultra-detailed, ",
    negative_prompt="cartoon, low quality, sharp edges, artifacts, ugly distortion, collage, cropped, cut off, " \
    "out of frame, bizarre, deformed, mutated, extra limbs, missing limbs, disfigured, poorly drawn, " \
    "asymmetrical, unrealistic, cartoonish, low resolution, pixelated, jpeg artifacts",
    strength=0.35,
    denoise_steps_coef=1.0,
    guidance_scale=7)

inpainted_image.save("inpainted_result.png")
inpainted_image.show()