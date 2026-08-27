import math
import torch
from diffusers import AutoencoderKL, StableDiffusionXLInpaintPipeline
from composer import Composer, CompositionResult
from PIL import Image
from diffusers import DPMSolverMultistepScheduler

class Inpainter:
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

    def inpaint(
        self,
        composition: CompositionResult = None,
        background: Image.Image = None,
        figure: Image.Image = None,
        prompt: str = "seamless integration, high quality, realistic lighting",
        negative_prompt: str = "blurry, low quality, sharp edges, artifacts",
        strength: float = 0.65,
        denoise_steps_coef: float = 1.0,
        guidance_scale: float = 7.5,
        seed: int = None
    ) -> Image.Image:
        if composition is not None:
            collage = composition.collage
            mask = composition.mask
        elif background is not None and figure is not None:
            comp_res = self._default_composer.compose(background, figure)
            collage = comp_res.collage
            mask = comp_res.mask
        else:
            raise ValueError("Передайте либо результат работы Composer, либо пару 'background' и 'figure'!")

        base_steps = 25
        num_inference_steps = max(1, math.ceil(base_steps * denoise_steps_coef))

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        if self.device == "mps":
            torch.mps.empty_cache()

        print(f"[Inpainter] Генерация (Steps: {num_inference_steps}, Strength: {strength})...")
        
        with torch.inference_mode():
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=collage.convert("RGB"),
                mask_image=mask.convert("L"),
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=collage.height,
                width=collage.width
            ).images[0]

        return output