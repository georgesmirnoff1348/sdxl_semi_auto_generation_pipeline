import torch
from diffusers import AutoencoderKL, StableDiffusionXLInpaintPipeline
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
import math
import cv2
import numpy as np
import random

class CenzorInpainter:
    def __init__(
        self, 
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    ):
        self.device = "mps" if torch.mps.is_available() else "cuda"
        print(f"--- СИСТЕМА ЦЕНЗОР: ПЕРЕХОД НА ЭВМ {self.device.upper()}...")
        
        self.dtype = torch.float16

        print(f"--- СИСТЕМА ЦЕНЗОР: ЗАГРУЗКА ВАРИАЦИОННОГО АВТОКОДЕРА... ---")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        print(f"--- СИСТЕМА ЦЕНЗОР: ЗАГРУЗКА ОСНОВНОЙ МОДЕЛИ {model_id.upper()}... ---")
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
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

    def inpaint_back(
        self,
        figure: Image.Image,
        alpha_print: Image.Image,
        back_object: str = "simple background",
        negative_prompt: str = None,
        inner_pad: int = 20,
        strength: float = 1.0,
        denoise_steps_coef: float = 1.0,
        guidance_scale: float = 7.5,
        seed: int = None,
        ) -> Image.Image:

        # 1. Извлекаем маску из alpha-канала или оттенков серого
        if alpha_print.mode in ("RGBA", "LA"):
            mask_np = np.array(alpha_print.split()[-1])
        else:
            mask_np = np.array(alpha_print.convert("L"))

        # 2. Бинаризация и эрозия (сжимаем силуэт персонажа внутрь)
        _, binary = cv2.threshold(mask_np, 128, 255, cv2.THRESH_BINARY)
        kernel_size = inner_pad * 2 + 1
        kernel_inner = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        eroded = cv2.erode(binary, kernel_inner, iterations=1)

        # 3. Инвертируем: белым (255) станет всё, кроме уменьшенной фигуры
        final_mask_np = cv2.bitwise_not(eroded)
        mask_image = Image.fromarray(final_mask_np)

        # 4. Настройка генератора случайных чисел
        if seed is None:
            seed = random.randint(0, 2147483647)
            print(f"🎲 Используется SEED: {seed}")
        else: print(f"Используется заранее заданный SEED: {seed}")

        generator = torch.Generator(device="cpu").manual_seed(seed)

        # 5. Запуск инпейнтинга (SDXL Pipeline)
        base_steps = 20
        num_inference_steps = max(1, math.ceil(base_steps * denoise_steps_coef))

        prompt = f"Background photography of a {back_object}, 1980s, casual photo."
        print(f"--- СИСТЕМА ЦЕНЗОР: ЗАПУСК ПОДСИСТЕМЫ ДОПОЛНЕНИЯ ДАННЫХ О МЕСТОПОЛОЖЕНИИ ЧЕЛОВЕЧЕСКОГО СУБЪЕКТА ---")
        print(f"--- СИСТЕМА ЦЕНЗОР: ИСПОЛЬЗУЮТСЯ ДАННЫЕ О МЕСТЕ {prompt}")
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=figure.convert("RGB"),
            mask_image=mask_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
    