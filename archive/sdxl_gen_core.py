import torch
import time
from diffusers import StableDiffusionXLPipeline
#from diffusers import EulerDiscreteScheduler
from diffusers import DPMSolverMultistepScheduler
#from diffusers import EulerAncestralDiscreteScheduler

model_default = "stabilityai/stable-diffusion-xl-base-1.0"
model1 = "SG161222/RealVisXL_V4.0" #реалистичные вхс фото

class CenzorGeneratorHealthyDPM:
    def __init__(self):
        print("--- ИНИЦИАЛИЗАЦИЯ ДЕТЕРМИНИРОВАННОГО ЯДРА К.О.Н.Т.У.Р. ---")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        model_id = model1
        # Загрузка пайплайна
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, 
            variant="fp16"
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config( 
        self.pipeline.scheduler.config,
        use_karras_sigmas = True #включаем сигмы Карраса для ускорения генерации 
        )

        self.pipeline.scheduler.algorithm_type = "dpmsolver++"
        self.pipeline = self.pipeline.to(self.device)
        print("✅ Модель успешно загружена в ОЗУ")
        print("Используется алгоритм генерации: DPM Solver Multistep Scheduler")
        print("Используется устройство: ", self.device)

    def generate(
            self, 
            prompt: str, 
            prompt_2: str = None, 
            negative_prompt: str = "", 
            num_inference_steps: int = 20, 
            guidance_scale: float = 7.5, 
            output_name: str = "output.png"):
        start_gen = time.time()
        
        image = self.pipeline(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=1024,
            height=1024
        ).images[0]
        
        image.save(output_name)
        print(f"Сохранено как '{output_name}' за {time.time() - start_gen:.2f} сек.")
        return image

class CenzorGeneratorInfectedDPM:
    def __init__(self):
        print("--- ИНИЦИАЛИЗАЦИЯ СТОХАСТИЧЕСКОГО ЯДРА К.О.Н.Т.У.Р. ---")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        model_id = model1
        # Загрузка пайплайна
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, 
            variant="fp16"
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config( 
        # DPM должен дать больше контраста и резкости, чтобы подчеркнуть "инфицированность" лица
        self.pipeline.scheduler.config,
        use_karras_sigmas = True #включаем сигмы Карраса для ускорения генерации 
        )
        self.pipeline.scheduler.algorithm_type = "sde-dpmsolver++"
        self.pipeline = self.pipeline.to(self.device)
        print("✅ Модель успешно загружена в ОЗУ")
        print("Используется алгоритм генерации: Stochastic DPM Solver Multistep Scheduler")
        print("Используется устройство: ", self.device)

    def generate(
                self, 
                prompt: str, 
                prompt_2: str = None, 
                negative_prompt: str = "", 
                num_inference_steps: int = 20, 
                guidance_scale: float = 7.5, 
                output_name: str = "output.png"):
            start_gen = time.time()
            
            image = self.pipeline(
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=1024,
                height=1024
            ).images[0]
            
            image.save(output_name)
            print(f"Сохранено как '{output_name}' за {time.time() - start_gen:.2f} сек.")
            return image