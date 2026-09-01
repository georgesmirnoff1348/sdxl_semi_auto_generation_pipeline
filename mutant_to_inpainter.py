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

ages = ("young", "adult", "old")
genders = ("man", "woman")
age = random.choice(ages)
gender = random.choice(genders)

# CLIP ViT-L: Локализуем аномалию СТРОГО на коже лица
prompt_o41 = (
    f"cursed creepy photo of an affected {age} {gender}, "
    f"face skin made of rough porous grey concrete, "  # skin made of вместо covered in
    f"petrified skin texture with deep dry fissures and crumbling cement pores, "
    f"distorted grin, unnaturally stretched mouth, unnaturally big eyes, uncanny valley"
)
# OpenCLIP ViT-bigG: Стилизация под архивную съемку
prompt2_o41 = (
    f"1980s analogue horror photo of a {age} {gender}, "
    f"disturbing medical archive photograph, "
    f"calcified grey stone face, calcification of human tissue, matte raw concrete material"
)
# Негативный промпт (с правильными запятыми и пробелами!)
negative_o41 = (
    "dirt, mud, soot, wet, glossy, makeup, paint, liquid, "  # блокируем грязь и мази
    "tongue, mushroom cap, hat, helmet, 3d render, sculpture, statue, nudity, naked, "
    "normal skin, smooth skin, beauty, cute, illustration, "
    "monochrome, black and white, extra limbs"
)
mutagen.generate_mutant(prompt_o41=prompt_o41,
                        #prompt2_o41=prompt2_o41,
                        negative_o41=negative_o41,
                        output_name=mutants_dir / mutant_filename,
                        num_inference_steps=20)
print(f"--- СИСTЕМА ЦЕНЗОР: ПРОИЗВЕДЕНО ИЗВЛЕЧЕНИЕ В ЭЛЕКТРОННУЮ БАЗУ ПОД ИМЕНЕМ {mutants_dir / mutant_filename}")

#2 Дорисовка фона -- не нужна, на самом деле, поэтому комментируем полностью
"""
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
                    back_object = "medical room, painted wall background, cracking paint",
                    negative_prompt= "people, illustration, human, outdoor", 
                    inner_pad=40,
                    strength=1.0,
                    denoise_steps_coef=1.0,
                    guidance_scale=7.5
                    ).save(inpainted_dir/inpback_filename)
"""