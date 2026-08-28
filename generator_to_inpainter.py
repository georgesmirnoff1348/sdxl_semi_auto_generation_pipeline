#0 Генерация фона (будет отменено в итоговом пайпе)
#Подключение библиотек и создание имен, создание экземпляра генератора
from pathlib import Path
from PIL import Image
from numberedfilesaver import get_next_available_filename

back_dir = Path("gen/backgrounds")
back_dir.mkdir(parents=True, exist_ok=True) # Создаст папку, если ее нет
back_prefix = "back_"

"""
from sdxl_gen_core import CenzorGeneratorDPM
citygen = CenzorGeneratorDPM()

backname = get_next_available_filename(directory= back_dir, prefix= back_prefix)
"""
import random
places = (
    "quiet soviet street view",
    "soviet bus stop",
    "soviet factory",
    "soviet khrushchyovka building",
    "empty rusted soviet playground, simple metal swing",
    "soviet hospital entrance",
    "empty soviet courtyard with concrete fence",
    "soviet boiler house with high chimney",
    #"pedestrian alley between panel buildings",
    "deserted tram stops and tracks"
)

prompt_city = (f"""
    architectural photography of a {random.choice(places)}, 1980s,
    off-center composition, just background
""")

negative_prompt_city = ("""
    centered composition, ground, winter, snow, distorted architecture, text, 
    warped, destroyed, ruins, aerial view, top-down, low-angle, anime, 
    illustration, painting, text, skyscrapers, high-rise
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

#1 Рисуем чела на фон инпейнтом
from cenzor_inpainter import CenzorInpainter
inpainter = CenzorInpainter()
composition = "close-up" #random.choice(("close-up", "half-body"))

inpainted_dir = Path("gen/inpaints")
inpainted_dir.mkdir(parents=True, exist_ok=True) # Создаст папку, если ее нет
inphuman_filename = get_next_available_filename(directory=inpainted_dir, prefix="inphuman_")
background_file = random.choice(list(back_dir.glob("*.png")))

inpainter.inpaint_human(background=Image.open(background_file),
                        composition="close-up",
                        age="young", 
                        gender="man",
                        nationality="georgian", 
                        clothing="worker suit",
                        strength=0.8,
                        denoise_steps_coef=1.0,
                        guidance_scale=10.0
                        ).save(inpainted_dir/inphuman_filename)