#1 Генерация мутанта
#Подключение библиотек и создание имен, создание экземпляра генератора
from pathlib import Path
from PIL import Image
from numberedfilesaver import get_next_available_filename
from ai_basic_alina.mutagen_old import MutaGen
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
    f"1980s portrait photo of a {age} {gender}, "
    f"thick rough grey concrete crust growing on face skin, "
    f"cracked grey cement scaling on face, "
    f"creepy gaze, flash photography, concrete background"
)
negative_o41 = (
    "statue, bust, sculpture, outdoor, 3d render, CGI, mannequin, "
    "peeling paint, makeup, paper, plaster, beauty, close-up, "
    "drawn, cartoon, illustration, monochrome, dark background, "
    "war paint, stripes, hood, arch, frame, border, framing, vignette"
)     

mutagen.generate_mutant(prompt_o41=prompt_o41,
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