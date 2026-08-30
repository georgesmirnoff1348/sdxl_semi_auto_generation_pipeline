import torch, time
from sdxl_gen_core import CenzorGeneratorDPM
from pathlib import Path

class MutaGen:
    def __init__(self):
        print("--- ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ ИЗВЛЕЧЕНИЯ МУТАНТНЫХ ОБРАЗОВ ---")
        self.mutant_generator = CenzorGeneratorDPM(crooked=True)
        print("✅ Система генерации успешно инициализирована.")

    def generate_mutant(self,
                    prompt_o41: str,
                    prompt2_o41: str,
                    negative_o41: str,
                    output_name: str | Path = "output_mut.png",
                    num_inference_steps: int = 20,
                    guidance_scale: float = 5.0,
                    seed: int = None):
        print("--- СИСТЕМА ЦЕНЗОР: ИЗВЛЕЧЕНИЕ АНОМАЛЬНЫХ ДАННЫХ ИЗ БАЗ К.О.Н.Т.У.Р. ---")

        self.mutant_generator.generate(prompt=prompt_o41, 
                                        prompt_2=prompt2_o41, 
                                        negative_prompt=negative_o41,
                                        num_inference_steps=num_inference_steps,
                                        guidance_scale=guidance_scale,
                                        output_name=output_name,
                                        seed = seed)