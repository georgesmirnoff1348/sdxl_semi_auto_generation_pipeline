#Подключение библиотек и создание имен, создание экземпляра генератора
from pathlib import Path

back_dir = Path("gen/backgrounds")
back_dir.mkdir(parents=True, exist_ok=True) # Создаст папку, если ее нет
back_prefix = "back_"

from sdxl_gen_core import CenzorGeneratorDPM
healthygen = CenzorGeneratorDPM()

#0 Генерация фона (будет отменено в итоговом пайпе)
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

healthygen.generate(
    prompt=prompt_city,
    negative_prompt=negative_prompt_city,
    output_name=str(back_dir / backname),
    num_inference_steps=25,
    guidance_scale=7.5
)