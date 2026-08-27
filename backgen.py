from sdxl_gen_core import CenzorGeneratorDPM
import torch
import gc
import time

print("--- СИСТЕМА ЦЕНЗОР: ЗАПУСК ИЗВЛЕЧЕНИЯ ГОРОДСКИХ ФОТО ---")

opencity = CenzorGeneratorDPM()

building_type = "5-story khrushchyovka apartment blocks"
building_types = (
    "5-story khrushchyovka apartment blocks",
    "industrial factory buildings",
    "soviet bus station",
    "soviet park with benches and trees"
)
location = "industrial zone"
locations = (
    'brutalist city center', 
    'industrial zone', 
    'residential area', 
    'park'
)

prompt_city = (f"""
An eye-level architectural photorealistic shot of a soviet {building_type} in {location}. 
Straight vertical lines, symmetrical composition.
"""
)

prompt2_city = (f"""
USSR 1980s {location} view, grey brutalist {building_type}, 
street, soviet architecture, weathered facades, overcast lighting, 
photorealistic, professional architecture photography
"""
)

negative_prompt_city = ("""
winter, snow, distorted architecture, curved walls, warped lines, tilted horizon, leaning buildings,
ruins, completely destroyed buildings, post-apocalyptic,
distant view, aerial view, birds-eye view, top-down view, low-angle view,
cartoon, anime, illustration, drawing, painting,
unrealistic, deformed, poster, cars, text,
skyscrapers, high-rise buildings, towers
"""
)
opencity.generate(prompt=prompt_city,
    prompt_2=prompt2_city,
    negative_prompt=negative_prompt_city,
    num_inference_steps=25,
    guidance_scale=5,
    output_name=f"back1.png"
    )
torch.mps.empty_cache()
gc.collect()
print ("⏳ Пауза 30 секунд для охлаждения...")
time.sleep(30)