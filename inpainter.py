import math
import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline
from composer import Composer, CompositionResult


class ImageInpainter:
    def __init__(
        self, 
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        device: str = "mps",
        torch_dtype: torch.dtype = torch.float16
    ):
        self.device = device
        self.dtype = torch_dtype
        
        # Немой дефолтный композер для автоматического режима
        self._default_composer = Composer(verbose=False)

        print(f"[Inpainter] Загрузка модели {model_id} на {device}...")
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            variant="fp16"
        ).to(self.device)


    def inpaint(
        self,
        composition: CompositionResult = None,
        background: Image.Image = None,
        figure: Image.Image = None,
        prompt: str = "seamless integration, high quality, realistic lighting, soft shadows",
        negative_prompt: str = "blurry, low quality, sharp edges, artifacts, ugly distortion",
        strength: float = 0.65,
        denoise_power: float = 1.0,
        guidance_scale: float = 7.5,
        seed: int = None
    ) -> Image.Image:
        """
        Метод гармонизации изображения. Принимает либо готовый CompositionResult,
        либо пару (background, figure) для автоматической немой сборки.
        """
        # 1. Извлечение коллажа и маски в зависимости от переданных аргументов
        if composition is not None:
            collage = composition.collage
            mask = composition.mask
        elif background is not None and figure is not None:
            comp_res = self._default_composer.compose(background, figure)
            collage = comp_res.collage
            mask = comp_res.mask
        else:
            raise ValueError("Передайте либо результат работы Composer, либо парами 'background' и 'figure'!")

        # 2. Определяем количество шагов диффузии в зависимости от силы денойза
        base_steps = 20
        num_inference_steps = max(1, math.ceil(base_steps * denoise_power))  
        # Увеличиваем шаги при большей мощности денойза

        # 3. Фиксация генератора случайных чисел (Seed)
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # 4. Запуск диффузионного пайплайна SDXL Inpaint
        print(f"[Inpainter] Генерация (Steps: {num_inference_steps}, Strength: {strength})...")
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=collage,
            mask_image=mask,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        return output