from diffusors_core import FactorDiffusor
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from configs import FactorPrompts, FactorInferenceParameters
from PIL import Image
from pathlib import Path
from typing import Optional
from dataclasses import asdict
import random
import gc


class OrdinaryGen(FactorDiffusor):
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

    def __enter__(self):
        return self

    def generate_image(
        self, 
        prompts: FactorPrompts, 
        config: FactorInferenceParameters, 
        save_path: Optional[Path] = None
        ):
        prompts_dict = asdict(prompts)
        config_dict = asdict(config)

        actual_seed = (
            config_dict["seed"]
            if config_dict.get("seed") is not None
            else random.randint(0, 2147483647)
        )

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

    def unload(self) -> None:
        print(
            "--- СИСТЕМА ФАКТОР: НАЧАТО ИЗВЛЕЧЕНИЕ МОДЕЛИ ИЗ ОПЕРАТИВНОЙ ПАМЯТИ ---"
        )

        if hasattr(self, "pipeline"):
            del self.pipeline

        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif self.device == "mps":
            torch.mps.empty_cache()

        print(
            "--- СИСТЕМА ФАКТОР: ОПЕРАТИВНАЯ ПАМЯТЬ УСПЕШНО ОСВОБОЖДЕНА ---"
        )

    def __exit__(self):
        self.unload()

