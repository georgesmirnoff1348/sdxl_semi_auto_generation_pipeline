import torch
import time
import random
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler
from pathlib import Path

model0 = "stabilityai/stable-diffusion-xl-base-1.0"
model1 = "SG161222/RealVisXL_V4.0" #реалистичные фото

class CenzorGeneratorDPM:
    def __init__(self, crooked=False, lora = None, lora_weight = None, device = "mps"):
        cenzorkerneltype = "СТОХАСТИЧЕСКОГО" if crooked else "ДЕТЕРМИНИРОВАННОГО"
        print(f"--- ИНИЦИАЛИЗАЦИЯ {cenzorkerneltype} ЯДРА К.О.Н.Т.У.Р. ---")
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"
        else: 
            self.device = "cpu"
        
        model_id = model1
        # Загрузка пайплайна
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            dtype=torch.float16, 
            variant="fp16"
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config( 
        self.pipeline.scheduler.config,
        use_karras_sigmas = True #включаем сигмы Карраса для ускорения генерации 
        )
        if crooked:
            self.pipeline.scheduler.algorithm_type = "sde-dpmsolver++"
        else:
            self.pipeline.scheduler.algorithm_type = "dpmsolver++"

        if lora and lora_weight:
            print(f"🔗 Подключение LoRA: {lora}")
            self.pipeline.load_lora_weights(lora, adapter_name="single_lora")
            self.pipeline.set_adapters(["single_lora"], adapter_weights=[lora_weight])

        self.pipeline = self.pipeline.to(self.device)
        print("✅ Модель успешно загружена в ОЗУ")
        print("Используется алгоритм генерации: DPM Solver Multistep Scheduler")
        print("Используется устройство: ", self.device)
        self.pipeline.enable_attention_slicing()




    def generate(
            self, 
            prompt: str, 
            prompt_2: str = None, 
            negative_prompt: str = "", 
            num_inference_steps: int = 20, 
            guidance_scale: float = 7.5, 
            width: int = 1024,
            height: int = 1024,
            seed: int = None,
            output_name: str | Path = "output.png"):
        start_gen = time.time()

        # Если seed не указан, генерируем случайный
        if seed is None:
            seed = random.randint(0, 2147483647)
        print(f"🎲 Используется SEED: {seed}")
        generator = torch.Generator(device="cpu").manual_seed(seed)

        out_path = Path(output_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image = self.pipeline(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height
        ).images[0]
        image.save(out_path)
        print(f"Сохранено как '{output_name}' за {time.time() - start_gen:.2f} сек.")
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
        print("✅ Очистка кэша завершена. Память освобождена.")
        return image, seed