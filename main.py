from humangen_sporous import SporeGen

hgs = SporeGen()

prompt = ("photograph of a mutated soviet woman with gigantic, unnaturally bulging eyes, exophthalmic, "
    "hyper-detailed bulging eyeballs, heavy wrinkles, grey skin, cracked and dry skin, "
    "grotesque and eerie appearance, cracked stone-like skin texture with mold, concrete background, "
    "1980s aesthetic, photorealistic, vintage studio photo, film grain")

#prompt = "vintage horror portrait, giant bulging eyes, cracked dry skin, 1980s photo, film grain"
negative_prompt = ("hat, green, smooth skin, beauty, clean, makeup, 3d render, anime, plastic, skeleton, "
            "blurry, illustration, drawing, cartoon, green, frame, torn corners, crumpled edges")

hgs.generate_sporous(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_name="sporous_output.png",
    num_inference_steps=20,
    guidance_scale=7.5
)