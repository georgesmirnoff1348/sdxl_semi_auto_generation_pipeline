#1 Генерация персонажа
#Подключение библиотек и создание имен, создание экземпляра генератора
from pathlib import Path
from PIL import Image
from numberedfilesaver import get_next_available_filename
from humangen import HealthyGen

hgen = HealthyGen()
comrades_dir = Path("gen/comrades")
comrades_dir.mkdir(parents=True, exist_ok=True) # Создаст папку, если ее нет
comrade_filename = get_next_available_filename(directory=comrades_dir, prefix="comrade_")
print ("--- СИСТЕМА ЦЕНЗОР: ВЫГРУЗКА В ЭЛЕКТРОННУЮ БАЗУ ---")
print (f"--- СИСТЕМА ЦЕНЗОР: ВЫБРАНА ДИРЕКТОРИЯ {comrades_dir} ---")
print (f"--- СИСТЕМА ЦЕНЗОР: ИМЯ ФАЙЛА {comrade_filename} ---")

hgen.generate_healthy(age="adult",
                      gender="man",
                      nationality="georgian",
                      clothing="simple shirt",
                      composition="half-body photo",
                      output_name=comrades_dir / comrade_filename)
print(f"--- СИСTЕМА ЦЕНЗОР: ПРОИЗВЕДЕНО ИЗВЛЕЧЕНИЕ В ЭЛЕКТРОННУЮ БАЗУ ПОД ИМЕНЕМ {comrades_dir / comrade_filename}")

#2 Дорисовка фона
import random
from cenzor_inpainter import CenzorInpainter
inpainter = CenzorInpainter()

places = (
    "quiet soviet street",
    "soviet bus stop",
    "soviet factory",
    "soviet khrushchyovka building",
    "empty rusted soviet playground, simple metal swing",
    "soviet hospital entrance",
    "empty soviet courtyard with concrete fence",
    "soviet boiler house",
    "panel buildings",
    "deserted tram stops and tracks"
)

inpainted_dir = Path("gen/inpaints")
inpainted_dir.mkdir(parents=True, exist_ok=True) # Создаст папку, если ее нет
inpback_filename = get_next_available_filename(directory=inpainted_dir, prefix="inpback_")

comrade_number = Path(comrade_filename).stem.split("_")[-1]
cut_dir = Path("gen/cuts")
cut_dir.mkdir(parents=True, exist_ok=True) # Создаст папку, если ее нет

print("--- СИСТЕМА ЦЕНЗОР: АВТОМАТИЧЕСКОЕ %335мвУКВЫР,% ОБРАЗА ЧЕЛОВЕЧЕСКОГО ЛИЦА ---")
from cutter import remove_background
remove_background(
    picture_path = comrades_dir / comrade_filename,
    output_path=cut_dir / f"comr_noback_{comrade_number}.png",
)
print("--- СИСТЕМА ЦЕНЗОР: №№:%:,% ЛИЦА ПРОИЗВЕДЕНО ---")

inpainter.inpaint_back(figure=Image.open(comrades_dir / comrade_filename),
                    alpha_print=Image.open(cut_dir / f"comr_noback_{comrade_number}.png"),
                    back_object = random.choice(places),
                    strength=1.0,
                    denoise_steps_coef=1.0,
                    guidance_scale=7.5
                    ).save(inpainted_dir/inpback_filename)
