#1 Генерация мутанта
#Подключение библиотек и создание имен, создание экземпляра генератора
from pathlib import Path
from PIL import Image
from numberedfilesaver import get_next_available_filename
from mutagen import MutaGen
import random

mutagen = MutaGen()
mutants_dir = Path("gen/mutants")
mutants_dir.mkdir(parents=True, exist_ok=True) # Создаст папку, если ее нет
mutant_filename = get_next_available_filename(directory=mutants_dir, prefix="mutant_")
print ("--- СИСТЕМА ЦЕНЗОР: ВЫГРУЗКА В ЭЛЕКТРОННУЮ БАЗУ ---")
print (f"--- СИСТЕМА ЦЕНЗОР: ВЫБРАНА ДИРЕКТОРИЯ {mutants_dir} ---")
print (f"--- СИСТЕМА ЦЕНЗОР: ИМЯ ФАЙЛА {mutant_filename} ---")

ages = ("young", "middle-aged", "elderly")
genders = ("man", "woman")
age = random.choice(ages)
gender = random.choice(genders)

prompt_o41 = (
    f"color portrait of a {age} {gender}, infected by cosmic strain"
    f"face skin with concrete texture"
)
prompt2_o41 = f"uncanny photograph of an infected {age} {gender} with concrete skin"
negative_o41 = (
    "normal looks, smooth skin, monochrome, zombie, beauty"
    "3d render, illustration, glossy, colorful, extra arms"
)

mutagen.generate_mutant(prompt_о41=prompt_o41,
                        prompt2_o41=prompt2_o41,
                        negative_o41=negative_o41,
                        output_name=mutants_dir / mutant_filename,
                        num_inference_steps=20)
print(f"--- СИСTЕМА ЦЕНЗОР: ПРОИЗВЕДЕНО ИЗВЛЕЧЕНИЕ В ЭЛЕКТРОННУЮ БАЗУ ПОД ИМЕНЕМ {mutants_dir / mutant_filename}")

#2 Дорисовка фона
from cenzor_inpainter import CenzorInpainter
inpainter = CenzorInpainter()

inpainted_dir = Path("gen/hosp_mutants")
inpainted_dir.mkdir(parents=True, exist_ok=True) # Создаст папку, если ее нет
inpback_filename = get_next_available_filename(directory=inpainted_dir, prefix="inpback_")

mutant_number = Path(mutant_filename).stem.split("_")[-1]
cut_dir = Path("gen/mut_cuts")
cut_dir.mkdir(parents=True, exist_ok=True) # Создаст папку, если ее нет

print("--- СИСТЕМА ЦЕНЗОР: АВТОМАТИЧЕСКОЕ %335мвУКВЫР,% ОБРАЗА ЧЕЛОВЕЧЕСКОГО ЛИЦА ---")
from cutter import remove_background
remove_background(
    picture_path = mutants_dir / mutant_filename,
    output_path=cut_dir / f"mut_noback_{mutant_number}.png",
)
print("--- СИСТЕМА ЦЕНЗОР: №№:%:,% ЛИЦА ПРОИЗВЕДЕНО ---")

inpainter.inpaint_back(figure=Image.open(mutants_dir / mutant_filename),
                    alpha_print=Image.open(cut_dir / f"mut_noback_{mutant_number}.png"),
                    back_object = "indoor medical room, painted wall background, cracking paint",
                    inner_pad=40,
                    strength=1.0,
                    denoise_steps_coef=1.0,
                    guidance_scale=7.5
                    ).save(inpainted_dir/inpback_filename)
