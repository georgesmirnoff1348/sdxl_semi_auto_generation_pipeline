from PIL import Image
from rembg import new_session, remove
from diffusors_core import FactorCutter
from pathlib import Path
from typing import Optional
from diffusors_core import factortimeinference
import torch, gc

class Cutter (FactorCutter):
    @factortimeinference
    def __init__(self, model_name: str = "u2net"):
        providers = []
        
        # 1. Если есть CUDA (например, Colab с T4/V100/A100)
        if torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")
            
        # 2. Если запускаем локально на macOS с чипом Apple Silicon
        if torch.mps.is_available():
            providers.append("MPSExecutionProvider")
            
        # 3. Дефолтный фоллбек на CPU
        providers.append("CPUExecutionProvider")

        self.session = new_session(model_name=model_name, providers=providers)

    @factortimeinference
    def remove_background(
            self, 
            image: Image.Image, 
            save_path: Optional[Path] = None):

        output_image = remove(image, session=self.session)

        if save_path is not None:
            output_image.save(save_path)

        return output_image

    def close(self):
        """Явно выгружает сессию rembg/ONNX и чистит VRAM."""
        if hasattr(self, 'session') and self.session is not None:
            del self.session
            self.session = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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


