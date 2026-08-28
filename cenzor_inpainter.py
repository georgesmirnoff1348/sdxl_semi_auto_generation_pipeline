import math
import torch
from diffusers import AutoencoderKL, StableDiffusionXLInpaintPipeline
from composer import Composer
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
from maskgen import generate_frame_mask

class CenzorInpainter:
    def __init__(
        self, 
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        device: str = "mps"
    ):
        self.device = device
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
        

        print(f"[Inpainter] Перевод моделей на {device}...")
        self.pipe.to(device)

        # Оставляем ТОЛЬКО slicing — tiling на MPS ломает память (contiguous stride)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.disable_tiling()

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
            generator = torch.Generator(device="cpu").manual_seed(seed)

        if self.device == "mps":
            torch.mps.empty_cache()

        print(f"[Inpainter] Генерация (Steps: {num_inference_steps}, Strength: {strength})...")

        # Первый текстовый кодер (CLIP ViT-L) — задает геометрию и базовую суть
        prompt_healthy = (
            f"""foreground subject, filling the scene, {composition} portrait 
            photo of a {age} {nationality} {gender}, wearing {clothing}."""
        )

        # Второй текстовый кодер (OpenCLIP ViT-bigG) — задает стиль, свет и детализацию
        prompt2_healthy = (
            f"1980s soviet archival document portrair photo, {composition} of a {age} {nationality} {gender}, "
            f"wearing {clothing}, natural skin texture, analogue film grain"
        )

        # Негативный промпт (дублируется или разделяется аналогично)
        negative_healthy = ("""
            deformed face, bad anatomy, military uniform, hat, legs, full-body, distant view, small human
            3d render, illustration, smooth plastic skin, low resolution, blurry
        """)

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