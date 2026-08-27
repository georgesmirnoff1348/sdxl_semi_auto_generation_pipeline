from PIL import Image
import torch
from transformers import pipeline

def remove_background(picture_path, output_path):
    pipe = pipeline(
        "image-segmentation", 
        model="briaai/RMBG-1.4", 
        trust_remote_code=True,
        device = torch.device ("mps") if torch.backends.mps.is_available() else "cpu"
    )

    image = Image.open(picture_path).convert("RGB")

    final_image = pipe(image)

    final_image.save(output_path)
    print(f"Фон в {picture_path} успешно удален!")