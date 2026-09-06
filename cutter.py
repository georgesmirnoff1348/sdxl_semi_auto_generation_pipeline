from PIL import Image
from rembg import new_session, remove
from diffusors_core import FactorCutter
from pathlib import Path
from typing import Optional
from diffusors_core import factortimeinference

class Cutter (FactorCutter):
    def __init__(self, cuttertype: str = "u2net"):
        providers = ["CUDAExecutionProvider", "MPSExecutionProvider", "CPUExecutionProvider"]
        self.session = new_session(cuttertype, providers=providers)

    @factortimeinference
    def remove_background(
            self, 
            image: Image.Image, 
            save_path: Optional[Path] = None):

        output_image = remove(image, session=self.session)

        if save_path is not None:
            output_image.save(save_path)

        return output_image


# Ярлык для U2net
class U2netCutter(Cutter):
    def __init__(self):
        # Прокидываем название модели в конструктор BaseRembgCutter через super()
        super().__init__(model_name="u2net")


# Ярлык для BiRefNet
class BirefNetCutter(Cutter):
    def __init__(self):
        # То же самое для birefnet
        super().__init__(model_name="birefnet-general")

    
"""
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = new_session("u2net", providers=providers)

def remove_background(picture_path: str, output_path: str):
    with Image.open(picture_path) as input_image:
        output_image = remove(input_image, session=session)
        output_image.save(output_path)
"""