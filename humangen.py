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
        output_num: int,
        seed: int = None,  # ◄◄◄ Можно передать свой сид или оставить None
    ):
        print(
            "--- СИСТЕМА ЦЕНЗОР: ИЗВЛЕЧЕНИЕ ЛИЦЕВЫХ ДАННЫХ ИЗ БАЗ К.О.Н.Т.У.Р. ---"
        )

        prompt_healthy = (
            f"A detailed waist-up portrait of a {age} {nationality} {gender}, "
            f"wearing {clothing}, distinct ethnic facial features, authentic eyes, "
            f"realistic skin texture, solid neutral studio background, 35mm photograph"
        )

        prompt2_healthy = (
            f"A waist-up shot of a {age} {nationality} {gender}, "
            f"1980s soviet archival document portrait, neutral serious expression, "
            f"soft studio lighting, sharp focus, analogue film grain, masterpiece photography"
        )

        antiprompt_healthy = (
            "asymmetry, close-up, extreme close-up, macro shot, cropped head, feet, "
            "wide shot, cap, hat, military uniform, ushanka, kgb, Soviet, communist, "
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
            guidance_scale=4.5,
            seed=seed,
            output_name=f"comrade_№{output_num}.png",
        )

        print(
            f"--- СИСТЕМА ЦЕНЗОР: ИЗВЛЕЧЕНА ЕДИНИЦА №{output_num} (SEED: {used_seed}) ---"
        )
        torch.mps.empty_cache()
        print("✅ Очистка кэша MPS завершена. Память освобождена.")