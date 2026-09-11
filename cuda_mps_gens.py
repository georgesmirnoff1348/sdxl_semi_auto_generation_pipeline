from diffusors_core import FactorDiffusor, FactorInpainter, factortimeinference
import torch
from diffusers import (StableDiffusionXLPipeline, 
                        DPMSolverMultistepScheduler, 
                        StableDiffusionXLInpaintPipeline,
                        AutoencoderKL,
                        ControlNetModel,
                        StableDiffusionXLControlNetInpaintPipeline)
from configs import FactorPrompts, FactorInferenceParameters
from PIL import Image
from pathlib import Path
from typing import Optional
from dataclasses import asdict
import random
import numpy as np
import cv2


class OrdinaryGen(FactorDiffusor):
    '''
    Class for generate everything -- just don't make the prompt too hard because it will compromise with quality
    '''
    def __init__(self, model: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"    
        print(f"--- ИНИЦИАЛИЗАЦИЯ ЯДРА СИСТЕМЫ Ф.А.К.Т.О.Р. НА ЭВМ ТИПА {self.device.upper()} ---")

        self.model = model
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model,
            dtype=torch.float16, 
            variant="fp16"
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config( 
            self.pipeline.scheduler.config,
            use_karras_sigmas = True #включаем сигмы Карраса для ускорения генерации 
        )
        self.pipeline.scheduler.algorithm_type = "dpmsolver++"
        self.pipeline = self.pipeline.to(self.device)
        self.pipeline.enable_attention_slicing()
        print("--- СИСТЕМА ФАКТОР: МОДЕЛЬ ИЗВЛЕЧЕНИЯ УСРЕДНЕННЫХ ОБРАЗОВ УСПЕШНО ЗАГРУЖЕНА В ОПЕРАТИВНУЮ ПАМЯТЬ ---")

    @factortimeinference
    def generate_image(
        self, 
        prompts: FactorPrompts, 
        config: FactorInferenceParameters, 
        save_path: Optional[Path] = None
        ):
        prompts_dict = asdict(prompts)
        config_dict = asdict(config)

        actual_seed = config_dict.pop("seed", None)

        # Implementing seed
        if actual_seed is None:
            actual_seed = random.randint(0, 2147483647)

        print(f"--- СИСТЕМА ФАКТОР: ИСПОЛЬЗУЕТСЯ ЯДРО СЛУЧАЙНОГО ЧИСЛА: {actual_seed} ---")
        del config_dict["seed"]
        config_dict["generator"] = torch.Generator(device="cpu").manual_seed(actual_seed)

        image = self.pipeline(
            **prompts_dict,
            **config_dict
        ).images[0]

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path)

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if self.device == "mps":
            torch.mps.empty_cache()

        return image


class BackInpainter(FactorInpainter):
    """
    Class was projected for inpainting only backgrounds,
    recieving raw studio photo + mask based on rmbg of character; 
    config extra shall contain params inner_pad, strength, mask (your rmbg'd character)
    """
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

    @factortimeinference
    def inpaint_image(self, 
                      prompts: FactorPrompts, 
                      config: FactorInferenceParameters, 
                      image: Image.Image, 
                      mask: Image.Image, 
                      save_path: Optional[Path] = None
                      ) -> Image.Image:
        alpha_print = mask
        prompts_dict = asdict(prompts)
        config_dict = asdict(config)
        inner_pad = config_dict.pop("inner_pad", 0)
        actual_seed = config_dict.pop("seed", None)

        # Implementing seed
        if actual_seed is None:
            actual_seed = random.randint(0, 2147483647)

        print(f"--- СИСТЕМА ФАКТОР: ИСПОЛЬЗУЕТСЯ ЯДРО СЛУЧАЙНОГО ЧИСЛА: {actual_seed} ---")
        config_dict["generator"] = torch.Generator(device="cpu").manual_seed(actual_seed)

        # Making true mask out of alpla-print
        if alpha_print.mode in ("RGBA", "LA"):
            mask_np = np.array(alpha_print.split()[-1])
        else:
            mask_np = np.array(alpha_print.convert("L"))

        # Eroding mask to harmonize character in backgroung properly
        _, binary = cv2.threshold(mask_np, 128, 255, cv2.THRESH_BINARY)
        kernel_size = inner_pad * 2 + 1
        kernel_inner = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        eroded = cv2.erode(binary, kernel_inner, iterations=1)

        # Inverting: mask is the place when we can draw
        final_mask_np = cv2.bitwise_not(eroded)
        config_dict["mask_image"] = Image.fromarray(final_mask_np)

        print(f"--- СИСТЕМА ЦЕНЗОР: ЗАПУСК ПОДСИСТЕМЫ ДОПОЛНЕНИЯ ДАННЫХ О МЕСТОПОЛОЖЕНИИ ЧЕЛОВЕЧЕСКОГО СУБЪЕКТА ---")
        print(f"--- СИСТЕМА ЦЕНЗОР: ИСПОЛЬЗУЮТСЯ ДАННЫЕ О МЕСТЕ {prompts_dict['prompt']}")

        image_inpainted = self.pipe(
            **prompts_dict,
            **config_dict,
            mask_image = mask,
            image=image
        ).images[0]

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            image_inpainted.save(save_path)

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if self.device == "mps":
            torch.mps.empty_cache()

        return image_inpainted


class ControlNetInpainter(FactorInpainter):
    """
    Inpaint class with ControlNet (any type, don't forget 
    to bring it in in configs as control_image)
    extra-params in config: strength: float,
            controlnet_conditioning_scale: float,
            control_guidance_end: float,
    """
    def __init__(
        self,
        base_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet_model_id: str = "xinsir/controlnet-tile-sdxl-1.0",
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
            use_safetensors=True
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
                use_karras_sigmas = True 
                )
        self.pipe.scheduler.algorithm_type = "dpmsolver++"

        # only slicing
        self.pipe.vae.enable_slicing()
        self.pipe.vae.disable_tiling()
        print(f"--- СИСТЕМА ЦЕНЗОР: ИНИЦИАЛИЗАЦИЯ КОНВЕЙЕРА ДОРИСОВКИ НА {self.device.upper()} ЗАВЕРШЕНА ---")

    @factortimeinference
    def inpaint_ControlNet(
        self,
        prompts: FactorPrompts,
        config: FactorInferenceParameters,
        image: Image.Image,               # for orig image
        alpha_print: Image.Image,         # to make a mask automatically
        control_image: Image.Image,       # controlnet reference
        save_path: Optional[Path] = None
    ) -> Image.Image:

        prompts_dict = asdict(prompts)
        config_dict = asdict(config)

        if alpha_print.mode in ("RGBA", "LA"):
            mask_np = np.array(alpha_print.split()[-1])
        else:
            mask_np = np.array(alpha_print.convert("L"))

        mask_np = cv2.GaussianBlur(mask_np, (15, 15), 0)
        mask_image = Image.fromarray(mask_np)

        actual_seed = config_dict.pop("seed", None)
        # Implementing seed
        if actual_seed is None:
            actual_seed = random.randint(0, 2147483647)
            
        print(f"--- СИСТЕМА ФАКТОР: ИСПОЛЬЗУЕТСЯ ЯДРО СЛУЧАЙНОГО ЧИСЛА: {actual_seed} ---")
        config_dict["generator"] = torch.Generator(device="cpu").manual_seed(actual_seed)


        print(f"--- СИСТЕМА ЦЕНЗОР: ЗАПУСК ПОДСИСТЕМЫ ДОПОЛНЕНИЯ ДАННЫХ О МЕСТОПОЛОЖЕНИИ ЧЕЛОВЕЧЕСКОГО СУБЪЕКТА ---")
        print(f"--- СИСТЕМА ЦЕНЗОР: ИСПОЛЬЗУЮТСЯ ДАННЫЕ: {prompts_dict['prompt'].upper()} ---")

        
        image_inpainted = self.pipe(
            **prompts_dict,
            **config_dict,
            image=image.convert("RGB"),
            mask_image=mask_image,
            control_image=control_image,
        ).images[0]

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            image_inpainted.save(save_path)

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if self.device == "mps":
            torch.mps.empty_cache()

        return image_inpainted


