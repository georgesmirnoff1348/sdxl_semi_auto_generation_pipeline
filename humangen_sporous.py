import time
import torch
from sdxl_gen_core import CenzorGeneratorDPM


class HealthyGen:
    def __init__(self):
        print(
            "--- ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ИЗВЛЕЧЕНИЯ АНОМАЛЬНЫХ ЧЕЛОВЕЧЕСКИХ ОБРАЗОВ ---"
        )
        self.normal_generator = CenzorGeneratorDPM(crooked=False)
        print("✅ Система генерации успешно инициализирована.")

    def generate_healthy(
        self,
        age: str,
        gender: str,
        nationality: str,
        composition: str,
        output_name: str,
        seed: int = None,
        num_inference_steps: int = 20,
    ):
        print("--- СИСТЕМА ЦЕНЗОР: ИЗВЛЕЧЕНИЕ АНОМАЛЬНЫХ ЛИЦЕВЫХ ДАННЫХ ИЗ БАЗ К.О.Н.Т.У.Р. ---")

        prompt_healthy = (
            f"A {composition} of a {age} {nationality} {gender}, very exhausted,  "
            f"concrete wall in the background, distinct ethnic facial features, authentic eyes, "
            f"very dry and cracked skin , 35mm photograph"
        )

        prompt2_healthy = (
            f"A photo of an exhausted {age} {nationality} {gender}, standing in front of concrete wall, "
            f"1980s soviet casual photo, very dry and cracked skin, "
            f"soft studio lighting, sharp focus, analogue film grain"
        )

        negative_healthy = (
            "distorted face, extreme close-up, macro shot, cropped head, military uniform"
            "siloviki, 3d render, anime, smooth plastic skin, blurry, crooked,"
            "digital artifacts, illustration, drawing, painting, unrealistic, cartoon, "
            "comic, deformed, poster, cars, flags, text"
        )
        print ("--- СИСТЕМА ЦЕНЗОР: ПОИСК В ФИЗИЧЕСКИХ БАЗАХ ПО ЗАПРОСУ ---")
        print (f"--- ЗАПРОС: {prompt_healthy} ---")
        
        # Пробрасываем seed в ядро
        _, used_seed = self.normal_generator.generate(
            prompt=prompt_healthy,
            prompt_2=prompt2_healthy,
            negative_prompt=negative_healthy,
            num_inference_steps=num_inference_steps,
            guidance_scale=5.0,
            seed=seed,
            output_name=output_name,
        )