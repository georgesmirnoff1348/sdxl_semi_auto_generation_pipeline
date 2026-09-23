
from cuda_mps_gens import ControlNetGen
from configs import FactorInferenceParameters, FactorPrompts
from pathlib import Path
from PIL import Image

prompts = FactorPrompts(
        prompt="a photorealistic scary creepy uncanny old man with exophtalmic severely bulging eyes and extremely wide grin smile, white hospital wall background",
        negative_prompt="drawing, 3d render, digial art, blurry, normal face, red eyes"
    )

config = FactorInferenceParameters(
    extra = {"controlnet_conditioning_scale": 0.8,
        "control_guidance_end": 0.7
    }
)

import PIL.ImageOps
# Загружаем набросок
input_sketch = Image.open("sporomanca.jpg").convert("L")

# Инвертируем: черные линии на белом -> белые линии на черном
control_image = PIL.ImageOps.invert(input_sketch).convert("RGB")

with ControlNetGen(model = "SG161222/RealVisXL_V4.0",
                   controlnet_model="xinsir/controlnet-scribble-sdxl-1.0"
                 ) as og:
    og.generate_image(config=config, 
                      prompts=prompts,
                      control_image=control_image,
                      save_path=Path(f"terminal_sporous.png"))


