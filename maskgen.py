import cv2
import numpy as np
import random
from PIL import Image

def generate_frame_mask(composition: str = "close-up", size: int = 1024) -> Image.Image:
    mask = np.zeros((size, size), dtype=np.uint8)
    
    if composition == "close-up":
        # --- БОЛЬШАЯ ТРАПЕЦИЯ ДЛЯ CLOSE-UP ---
        # Вариируем ширину верха (40-60%) и низа (70-90%)
        top_w = int(size * random.uniform(0.40, 0.60))
        bottom_w = int(size * random.uniform(0.70, 0.90))
        height = int(size * random.uniform(0.75, 0.90)) # занимает большую часть кадра
        
        center_x = size // 2 + int(size * random.uniform(-0.05, 0.05))
        
        pts = np.array([
            [center_x - top_w // 2, size - height],
            [center_x + top_w // 2, size - height],
            [center_x + bottom_w // 2, size],
            [center_x - bottom_w // 2, size]
        ], np.int32)
        
        cv2.fillPoly(mask, [pts], 255)
        
    elif composition == "half-body":
        # --- ПРЯМОУГОЛЬНИК ДЛЯ HALF-BODY ---
        rect_w = int(size * random.uniform(0.65, 0.90))
        rect_h = int(size * random.uniform(0.80, 0.95))
        
        x_left = random.randint(50, max(50, (size - rect_w)-50))
        
        cv2.rectangle(mask, (x_left, size - rect_h), (x_left + rect_w, size), 255, -1)
        
    else:
        raise ValueError(f"Неизвестный shot_type: {composition}")

    # Мягкий край для SDXL
    gb = random.choice((51,61,77,91))
    mask = cv2.GaussianBlur(mask, (gb, gb), 0)

    return Image.fromarray(mask)