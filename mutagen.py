import torch, time
from sdxl_gen_core import CenzorGeneratorDPM

class MutaGen:
    def __init__(self):
        print("--- ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ИЗВЛЕЧЕНИЯ МУТАНТНЫХ ОБРАЗОВ ---")
        self.crooked_generator = CenzorGeneratorDPM(crooked=True)
        print("✅ Система генерации успешно инициализирована.")

    def generate_muta(self, output_num: int):
        print("--- СИСТЕМА ЦЕНЗОР: ИЗВЛЕЧЕНИЕ АНОМАЛЬНЫХ ДАННЫХ ИЗ БАЗ К.О.Н.Т.У.Р. ---")
        torch.mps.empty_cache()
        print("✅ Очистка кэша MPS завершена. Память освобождена.")
        # Инициализируем ядро К.О.Н.Т.У.Р.а (веса загружаются в ОЗУ один раз)

        prompt_041 = (
            f'''A waist-up shot of 
            
            solid studio background, 
            sharp details, professional photography
            '''
        )

        # Во втором промпте заменяем "portrait" на конкретное описание плана
        prompt2_041 = (
            f'''

            '''
        )

        # Расширяем негативный промпт, добавляя явный запрет на крупные планы
        antiprompt_041 = ("""
        """)

        self.crooked_generator.generate(prompt=prompt_041, 
            prompt_2=prompt2_041, 
            negative_prompt=antiprompt_041,
            num_inference_steps=25,
            guidance_scale=5.5,
            output_name=f"monster_№{output_num}.png")
            # 1. Жестко очищаем кэш выделенной памяти MPS
        torch.mps.empty_cache()
            # 2. Даем Макбуку просто спокойно постоять и сбросить жар перед новой задачей
        print(f"⏳ Кадр готов. Аппаратная пауза 60 секунд для предотвращения троттлинга...")
        time.sleep(60)