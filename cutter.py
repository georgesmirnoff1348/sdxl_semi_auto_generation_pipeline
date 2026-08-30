from PIL import Image
from rembg import new_session, remove

# 1. Задаем явно CUDA провайдер
# 2. Инициализируем сессию ЕДИНОЖДЫ при импорте модуля
# (birefnet весит всего ~170MB, она не забьет RAM)
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = new_session("birefnet-general", providers=providers)

def remove_background(picture_path: str, output_path: str):
    with Image.open(picture_path) as input_image:
        output_image = remove(input_image, session=session)
        output_image.save(output_path)