from humangen_sporous import SporeGen

hgs = SporeGen()

prompt = ("photograph of a mutated soviet person with gigantic, unnaturally bulging eyes, "
    "hyper-detailed bulging pale green eyeballs, heavy wrinkles, "
    "cracked stone-like skin texture, patchy lichen and grime on the face, "
    "vintage tintype photograph style, "
    "eerie atmosphere, 1980s aesthetic, photorealistic")

#prompt = "vintage horror portrait, giant bulging eyes, cracked dry skin, 1980s photo, film grain"
negative_prompt = ("green, smooth skin, beauty, clean, makeup, 3d render, anime, plastic, skeleton, "
            "blurry, illustration, drawing, cartoon, green, frame")

hgs.generate_sporous(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_name="sporous_output.png",
    num_inference_steps=20,
    guidance_scale=5.0
)