import torch
from diffusers import AutoencoderKL, StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
import math
import cv2
import numpy as np
import random

class ControlNetCenzorInpainter:
    def __init__(
        self,
        base_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet_model_id: str = "diffusers/controlnet-depth-sdxl-1.0",
        vae_model_id: str = "madebyollin/sdxl-vae-fp16-fix",
    ):
        self.device = "mps" if torch.mps.is_available() else "cuda"
        print(f"--- СИСТЕМА ЦЕНЗОР: ПЕРЕХОД НА ЭВМ {self.device.upper()}...")
        
        self.dtype = torch.float16

        print(f"--- СИСТЕМА ЦЕНЗОР: ЗАГРУЗКА ВАРИАЦИОННОГО АВТОКОДЕРА... ---")
        vae = AutoencoderKL.from_pretrained(
            vae_model_id, 
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        print(f"--- СИСТЕМА ЦЕНЗОР: ЗАГРУЗКА КОНТРОЛЬНОЙ СЕТИ {controlnet_model_id.upper()}... ---")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.dtype == torch.float16 else None
        )

        print(f"--- СИСТЕМА ЦЕНЗОР: ЗАГРУЗКА ОСНОВНОЙ МОДЕЛИ {base_model_id.upper()} С КОНТРОЛИРУЮЩИМИ СЕТЯМИ ---")
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.dtype == torch.float16 else None
        )
        self.pipe.to(self.device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config( 
                self.pipe.scheduler.config,
                use_karras_sigmas = True #включаем сигмы Карраса для ускорения генерации 
                )
        self.pipe.scheduler.algorithm_type = "dpmsolver++"

        # Оставляем ТОЛЬКО slicing — tiling на MPS ломает память (contiguous stride)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.disable_tiling()
        print(f"--- СИСТЕМА ЦЕНЗОР: ИНИЦИАЛИЗАЦИЯ КОНВЕЙЕРА ДОРИСОВКИ НА {self.device.upper()} ЗАВЕРШЕНА ---")

    def inpaint_spores(
        self,
        image: Image.Image,               # Исходник с фоном
        alpha_print: Image.Image,          # Альфа-маска вырезанного человека
        depth_map: Image.Image,           # Карта глубин
        prompt: str,
        negative_prompt: str = "blurry, smooth skin, low quality, distortion",
        strength: float = 0.8,
        controlnet_scale: float = 0.55,
        guidance_scale: float = 7.5,
        denoise_steps_coef: float = 1.0,
        seed: int = None,
    ) -> Image.Image:

        # 1. Извлекаем маску из alpha-канала или оттенков серого
        if alpha_print.mode in ("RGBA", "LA"):
            mask_np = np.array(alpha_print.split()[-1])
        else:
            mask_np = np.array(alpha_print.convert("L"))

        # 2. Размытие маски для сглаживания краев
        mask_np = cv2.GaussianBlur(mask_np, (15, 15), 0)
        mask_image = Image.fromarray(mask_np)

        # 4. Seed на CPU (фиксит баг генератора MPS)
        if seed is None:
            seed = random.randint(0, 2147483647)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # 5. Запуск инпейнтинга (SDXL Pipeline)
        base_steps = 20
        num_inference_steps = max(1, math.ceil(base_steps * denoise_steps_coef))

        print(f"--- СИСТЕМА ЦЕНЗОР: ЗАПУСК ПОДСИСТЕМЫ ДОПОЛНЕНИЯ ДАННЫХ О МЕСТОПОЛОЖЕНИИ ЧЕЛОВЕЧЕСКОГО СУБЪЕКТА ---")
        print(f"--- СИСТЕМА ЦЕНЗОР: ИСПОЛЬЗУЮТСЯ ДАННЫЕ: {prompt.upper()} ---")

        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image.convert("RGB"),
            mask_image=mask_image,
            control_image=depth_map.convert("RGB"),
            strength=strength,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
    