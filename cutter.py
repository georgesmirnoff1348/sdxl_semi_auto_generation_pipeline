from PIL import Image
from rembg import remove, new_session

# Инициализируем сессию с моделью isnet-general-use (основа для RMBG) 
# или явно briaai (в свежих версиях rembg доступен и bria-rmbg)
def remove_background(picture_path: str, output_path: str):
    session = new_session("birefnet-general")

    input_image = Image.open(picture_path)
    output_image = remove(input_image, session=session)
    output_image.save(output_path)