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
        print(
            "--- СИСТЕМА ЦЕНЗОР: ИЗВЛЕЧЕНИЕ ЛИЦЕВЫХ ДАННЫХ ИЗ БАЗ К.О.Н.Т.У.Р. ---"
        )

        prompt_healthy = (
            f"A detailed {composition} of a {age} {nationality} {gender}, "
            f"wearing {clothing}, distinct ethnic facial features, authentic eyes, "
            f"realistic skin texture, solid neutral studio background, 35mm photograph"
        )

        prompt2_healthy = (
            f"A waist-up shot of a {age} {nationality} {gender}, "
            f"1980s soviet archival document portrait, neutral serious expression, "
            f"soft studio lighting, sharp focus, analogue film grain, masterpiece photography"
        )

        antiprompt_healthy = (
            "distorted face, extreme close-up, macro shot, cropped head, feet, "
            "wide shot, cap, hat, military uniform, ushanka, kgb, communist, "
            "military officer, police, visor cap, peaked cap, epaulets, medals, badge, "
            "siloviki, 3d render, anime, smooth plastic skin, blurry, crooked, "
            "digital artifacts, illustration, drawing, painting, unrealistic, cartoon, "
            "comic, deformed, poster, cars, flags, text"
        )

        # Пробрасываем seed в ядро
        _, used_seed = self.normal_generator.generate(
            prompt=prompt_healthy,
            prompt_2=prompt2_healthy,
            negative_prompt=antiprompt_healthy,
            num_inference_steps=25,
            guidance_scale=7.5,
            seed=seed,
            output_name=output_name,
        )
        torch.mps.empty_cache()
        print("✅ Очистка кэша MPS завершена. Память освобождена.")