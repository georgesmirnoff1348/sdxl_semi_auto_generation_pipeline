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
    # 2. Загружаем ваше оригинальное изображение
    image = Image.open(picture_path).convert("RGB")

    # 3. Нейросеть сама всё вырезает и возвращает ГОТОВУЮ картинку с прозрачным фоном
    final_image = pipe(image)

    # 4. Просто сохраняем готовый результат в формате PNG
    final_image.save(output_path)
    print(f"Фон в {picture_path} успешно удален!")