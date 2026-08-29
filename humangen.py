import time
import torch
from sdxl_gen_core import CenzorGeneratorDPM


class HealthyGen:
    def __init__(self):
        print(
            "--- ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ИЗВЛЕЧЕНИЯ ЧЕЛОВЕЧЕСКИХ ОБРАЗОВ ---"
        )
        self.normal_generator = CenzorGeneratorDPM(crooked=False)
        print("✅ Система генерации успешно инициализирована.")

    def generate_healthy(
        self,
        age: str,
        gender: str,
        nationality: str,
        clothing: str,
        composition: str,
        output_name: str,
        seed: int = None,
    ):
        print("--- СИСТЕМА ЦЕНЗОР: ИЗВЛЕЧЕНИЕ ЛИЦЕВЫХ ДАННЫХ ИЗ БАЗ К.О.Н.Т.У.Р. ---")

        prompt_healthy = (
            f"A {composition} of a {age} {nationality} {gender}, "
            f"wearing {clothing}, distinct ethnic facial features, authentic eyes, "
            f"realistic skin texture, solid neutral studio background, 35mm photograph"
        )

        prompt2_healthy = (
            f"A {composition} of a {age} {nationality} {gender}, wearing {clothing}"
            f"1980s soviet casual photo, "
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
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=seed,
            output_name=output_name,
        )