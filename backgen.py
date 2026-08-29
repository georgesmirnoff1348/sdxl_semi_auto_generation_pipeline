from sdxl_gen_core import CenzorGeneratorDPM
import torch
import gc
import time
import random

print("--- СИСТЕМА ЦЕНЗОР: ЗАПУСК ИЗВЛЕЧЕНИЯ ГОРОДСКИХ ФОТО ---")

opencity = CenzorGeneratorDPM()

places = (
    "quiet soviet street",
    "soviet bus stop",
    "soviet factory",
    "soviet khrushchyovka building",
    "empty rusted soviet playground, simple metal swing"
    "soviet hospital entrance"
    "empty soviet courtyard with concrete fence"
    "soviet boiler house with high chimney"
    "pedestrian alley between panel buildings"
    "deserted tram stops and tracks"
)



prompt_city = (f"""
    architectural photography of a {random.choice(places)}, 1980s,
    straight perspective
""")

negative_prompt_city = ("""
    winter, snow, distorted architecture, warped, destroyed, ruins,
    aerial view, top-down, low-angle, anime, illustration, painting,
    text, skyscrapers, high-rise
""")

opencity.generate(prompt=prompt_city,
    #prompt_2=prompt2_city,
    negative_prompt=negative_prompt_city,
    num_inference_steps=25,
    guidance_scale=7.5,
    output_name=f"back2.png"
    )
torch.mps.empty_cache()
gc.collect()
print ("⏳ Пауза 30 секунд для охлаждения...")
time.sleep(30)