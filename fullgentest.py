#0 Генерация фона (будет отменено в итоговом пайпе)
#Подключение библиотек и создание имен, создание экземпляра генератора
from pathlib import Path

back_dir = Path("gen/backgrounds")
back_dir.mkdir(parents=True, exist_ok=True) # Создаст папку, если ее нет
back_prefix = "back_"

from sdxl_gen_core import CenzorGeneratorDPM
citygen = CenzorGeneratorDPM()

from numberedfilesaver import get_next_available_filename
backname = get_next_available_filename(directory= back_dir, prefix= back_prefix)

import random
places = (
    "quiet soviet street",
    "soviet bus stop",
    "soviet factory",
    "soviet khrushchyovka building",
    "empty rusted soviet playground, simple metal swing",
    "soviet hospital entrance",
    "empty soviet courtyard with concrete fence",
    "soviet boiler house with high chimney",
    "pedestrian alley between panel buildings",
    "deserted tram stops and tracks"
)

prompt_city = (f"""
    architectural photography of a {random.choice(places)}, 1980s,
    straight perspective, 
""")

negative_prompt_city = ("""
    winter, snow, distorted architecture, warped, destroyed, ruins,
    aerial view, top-down, low-angle, anime, illustration, painting,
    text, skyscrapers, high-rise
""")

"""
citygen.generate(
    prompt=prompt_city,
    negative_prompt=negative_prompt_city,
    output_name=str(back_dir / backname),
    num_inference_steps=25,
    guidance_scale=7.5
)
"""

#1 Генерация чела
"""from humangen import HealthyGen
hg = HealthyGen()

comrade_dir = Path("gen/comrades")
comrade_dir.mkdir(parents=True, exist_ok=True) # Создаст папку, если ее нет
comrade_prefix = "comrade_"
comradename = get_next_available_filename(directory= comrade_dir, prefix= comrade_prefix)
hg.generate_healthy(age = "adult",
                    gender="man",
                    nationality="chechen",
                    clothing="simple worker shirt",
                    composition="half-body photo",
                    output_name= str(comrade_dir / comradename)
)
comrade_filename = comrade_dir / comradename
"""
comrade_filename = "comrade_4"

#2 Очистка от фона
from cutter import remove_background
remove_background(str(comrade_filename), str(comrade_filename))

#3 Размещение на фоне
from composer import Composer
comp = Composer()

composed_dir = Path("gen/photos")
composed_dir.mkdir(parents=True, exist_ok=True) # Создаст папку, если ее нет
clean_name = Path(comrade_filename).stem 
num = int(clean_name.split("_")[-1])

composedname = f"photo_{num}"
background_file = random.choice(list(back_dir.glob("*.png")))
comp.compose(background=str(background_file), 
            figure=str(comrade_filename),
            output_mode= "collage"
            ).save(composed_dir / composedname)