
from cuda_mps_gens import OrdinaryGen
from configs import FactorInferenceParameters, FactorPrompts
from pathlib import Path

prompts = FactorPrompts(
        prompt="analog horror film still, grotesque studio portrait, " \
        "horrifying wide bulging popped eyes, bug-eyes, unsettling pupils, " \
        "wide manic grin revealing irregular rotten teeth, disheveled hair, " \
        "harsh low lighting, grainy retro horror footage, raw skin texture",
        negative_prompt="smooth skin, symmetric, normal eyes, attractive, " \
        "digital art, cartoon, bright, clean, soft focus, green, zombie, monochrome"
    )

config = FactorInferenceParameters()

with OrdinaryGen(#model = "SG161222/RealVisXL_V4.0"
                 ) as og:
    og.generate_image(config=config, prompts=prompts, save_path=Path("bibaher5.png"))


