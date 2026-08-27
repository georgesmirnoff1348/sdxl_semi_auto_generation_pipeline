import torch, time
from sdxl_gen_core import CenzorGeneratorDPM

class HealthyGen:
    def __init__(self):
        print("--- ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ИЗВЛЕЧЕНИЯ ЧЕЛОВЕЧЕСКИХ ОБРАЗОВ ---")
        self.normal_generator = CenzorGeneratorDPM(crooked=False)
        print("✅ Система генерации успешно инициализирована.")

    def generate_healthy(self, age: str, 
                        gender: str, 
                        nationality: str, 
                        clothing: str, 
                        output_num: int):
        print("--- СИСТЕМА ЦЕНЗОР: ИЗВЛЕЧЕНИЕ ЛИЦЕВЫХ ДАННЫХ ИЗ БАЗ К.О.Н.Т.У.Р. ---")
        # Инициализируем ядро К.О.Н.Т.У.Р.а (веса загружаются в ОЗУ один раз)
        #normal = CenzorGeneratorDPM()

        prompt_healthy = (
            f'''A photograph of a {age} soviet {gender} standing straight, 
            {nationality} features, wearing a {clothing}, solid studio background, 
            photorealistic, sharp details, professional photography
            '''
        )
        prompt2_healthy = None

        # Расширяем негативный промпт, добавляя явный запрет на крупные планы
        antiprompt_healthy = ("""
        asymmetry, close-up, extreme close-up, macro shot, 
        cropped head, feet, wide shot, cap, hat, military uniform, 
        ushanka, kgb, Soviet, communist, military officer, police, 
        visor cap, peaked cap, epaulets, medals, badge, siloviki, 
        3d render, anime, smooth plastic skin, blurry, crooked,
        digital artifacts, illustration, drawing, painting, 
        unrealistic, cartoon, comic, deformed, poster, cars, flags, text,
        """)

        self.normal_generator.generate(prompt=prompt_healthy, 
            prompt_2=prompt2_healthy, 
            negative_prompt=antiprompt_healthy,
            num_inference_steps=25,
            guidance_scale=4.5,
            output_name=f"comrade_№{output_num}.png")
        print("--- СИСТЕМА ЦЕНЗОР: ИЗВЛЕЧЕНА ЕДИНИЦА ЛИЦЕВЫХ ДАННЫХ ИЗ БАЗ К.О.Н.Т.У.Р. ---")
        torch.mps.empty_cache()
        print("✅ Очистка кэша MPS завершена. Память освобождена.")