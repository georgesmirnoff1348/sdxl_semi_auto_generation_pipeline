from humangen import HealthyGen
from cutter import remove_background
import time
import torch

healthy_gen = HealthyGen()
for i in range(0):
    print("--- СИСТЕМА ЦЕНЗОР: СТАРТ ИЗВЛЕЧЕНИЯ ЛИЦА ---")
    healthy_gen.generate_healthy(age="middle-aged",
                                gender="man",
                                nationality='russian',
                                clothing='worker clothes',
                                output_num=i)

    remove_background(picture_path=f"comrade_№{i}.png",
                    output_path=f"comrade_№{i}.png")
    torch.mps.empty_cache()
    gc.collect()
    print ("⏳ Пауза 30 секунд для охлаждения...")
    time.sleep(30)