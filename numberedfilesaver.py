import re
from pathlib import Path

def get_next_available_filename(directory: str = "output", prefix: str = "picture_", extension: str =".png"):
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Регулярка для поиска чисел после префикса: back_(\d+).png
    pattern = re.compile(fr"^{re.escape(prefix)}(\d+){re.escape(extension)}$")
    
    existing_indices = set()
    
    # 1. Собираем все имеющиеся индексы
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            match = pattern.match(file_path.name)
            if match:
                existing_indices.add(int(match.group(1)))
    
    # 2. Ищем наименьший свободный номер, начиная с 1
    next_index = 1
    while next_index in existing_indices:
        next_index += 1
        
    # Возвращаем готовое имя файла (например, back_4.png)
    return f"{prefix}{next_index}{extension}"