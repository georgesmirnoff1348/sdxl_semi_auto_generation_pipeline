import math
import torch
from diffusers import AutoencoderKL, StableDiffusionXLInpaintPipeline
from composer import Composer
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
from maskgen import generate_frame_mask
import cv2
import numpy as np
import random

class CenzorInpainter:
    def __init__(
        self, 
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    ):
        
        print(f"[Inpainter] Перевод моделей на {self.device}...")
        self.device = "mps" if torch.mps.is_available() else "cuda"
        self.pipe.to(self.device)
        
        self._default_composer = Composer(verbose=False)

        self.dtype = torch.float16

        print("[Inpainter] Загрузка VAE...")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=self.dtype,
            use_safetensors=True
        )

        print(f"[Inpainter] Загрузка основной модели {model_id}...")
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.dtype == torch.float16 else None
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config( 
                self.pipe.scheduler.config,
                use_karras_sigmas = True #включаем сигмы Карраса для ускорения генерации 
                )
        self.pipe.scheduler.algorithm_type = "dpmsolver++"

        # Оставляем ТОЛЬКО slicing — tiling на MPS ломает память (contiguous stride)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.disable_tiling()
        print(
            "✅ [Inpainter] Подсистема дорисовывания успешно инициализирована.\n"
        )

    def inpaint_human(self,
        background: Image.Image,
        composition: str,
        age: str,
        gender: str,
        nationality: str,
        clothing: str,
        strength: float = 0.8,
        denoise_steps_coef: float = 1.0,
        guidance_scale: float = 7.5,
        seed: int = None
    ) -> Image.Image:

        mask = generate_frame_mask(composition = composition)

        if background is None or mask is None:
            raise ValueError("Передайте фон и процедурную маску!")

        base_steps = 25
        num_inference_steps = max(1, math.ceil(base_steps * denoise_steps_coef))

        generator = None
        if seed is not None:
            # На MPS генератор Diffusers должен быть на CPU, на CUDA можно явно указать CUDA
            gen_device = "cpu" if self.device == "mps" else self.device
            generator = torch.Generator(device=gen_device).manual_seed(seed)    

        if self.device == "mps":
            torch.mps.empty_cache()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        print(f"[Inpainter] Генерация (Steps: {num_inference_steps}, Strength: {strength})...")

        # CLIP ViT-L — геометрия, объект и тип съемки
        prompt_healthy = (
            f"{composition} photo of {age} {nationality} {gender}, wearing {clothing}, "
            f"waist-up portrait, medium shot, realistic skin, standing outdoors in front of a background"
        )

        # OpenCLIP ViT-bigG — стилистика (без слова document!)
        prompt2_healthy = (
            f"1980s soviet street portrait photography, film photo, {composition} of a {age} {nationality} {gender}, "
            f"wearing {clothing}, natural skin texture, soft daylight, analogue film grain"
        )

        # Негативный промпт — режем плакаты, рисунки и рамки
        negative_healthy = (
            "distant view, poster, painting, drawing, illustration, billboard, framed picture, sign, "
            "deformed face, bad anatomy, military uniform, hat, legs, feet, full-body, "
            "small human, 3d render, smooth plastic skin, low resolution, blurry"
        )
        with torch.inference_mode():
            output = self.pipe(
                prompt=prompt_healthy,
                prompt_2=prompt2_healthy,
                negative_prompt=negative_healthy,
                image=background.convert("RGB"),
                mask_image=mask.convert("L"),
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,

            ).images[0]

        return output

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
        print("--- СИСТЕМА ЦЕНЗОР: ЗАПУСК ПОДСИСТЕМЫ ДОПОЛНЕНИЯ ДАННЫХ О МЕСТОПОЛОЖЕНИИ ЧЕЛОВЕЧЕСКОГО СУБЪЕКТА ---")
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