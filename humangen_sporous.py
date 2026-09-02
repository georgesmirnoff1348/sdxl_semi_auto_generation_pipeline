from sdxl_gen_core import CenzorGeneratorDPM

class SporeGen:
    def __init__(self):
        print(
            "--- ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ИЗВЛЕЧЕНИЯ АНОМАЛЬНЫХ ЧЕЛОВЕЧЕСКИХ ОБРАЗОВ ---"
        )
        self.normal_generator = CenzorGeneratorDPM(crooked=False)
        print("✅ Система генерации успешно инициализирована.")

    def generate_sporous(
        self,
        prompt: str,
        prompt2: str = None,
        negative_prompt: str = None,
        output_name: str = None,
        seed: int = None,
        num_inference_steps: int = 20,
    ):
        print("--- СИСТЕМА ЦЕНЗОР: ИЗВЛЕЧЕНИЕ АНОМАЛЬНЫХ ЛИЦЕВЫХ ДАННЫХ ИЗ БАЗ К.О.Н.Т.У.Р. ---")

        prompt = prompt
        prompt2 = prompt2
        negative_prompt = negative_prompt

        print ("--- СИСТЕМА ЦЕНЗОР: ПОИСК В ФИЗИЧЕСКИХ БАЗАХ ПО ЗАПРОСУ ---")
        print (f"--- ЗАПРОС: {prompt} ---")
        
        # Пробрасываем seed в ядро
        _, used_seed = self.normal_generator.generate(
            prompt=prompt,
            prompt_2=prompt2,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=5.0,
            seed=seed,
            output_name=output_name,
        )