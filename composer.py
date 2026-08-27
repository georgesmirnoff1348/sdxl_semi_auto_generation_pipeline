import numpy as np
import cv2
from PIL import Image
from math import ceil
from dataclasses import dataclass

@dataclass
class CompositionResult:
    collage: Image.Image
    mask: Image.Image

class Composer:
    def __init__(self,
                 verbose: bool = True,
                 mask_inflate: int = 30,
                 interactive: bool = False):
        self.verbose = verbose
        self.mask_inflate = mask_inflate
        self.interactive = interactive

    def preview_ascii_layout(self,
                             background_size: tuple[int, int], 
                             figure_size: tuple[int, int], 
                             position: tuple[int, int], 
                             grid_width: int = 40):
        bg_w, bg_h = background_size
        fg_w, fg_h = figure_size
        pos_x, pos_y = position

        # 1. Определяем полные виртуальные границы (с учетом вылета за рамки)
        min_x = min(0, pos_x)
        min_y = min(0, pos_y)
        max_x = max(bg_w, pos_x + fg_w)
        max_y = max(bg_h, pos_y + fg_h)

        total_w = max_x - min_x
        total_h = max_y - min_y

        # 2. Высчитываем масштабирование под консоль
        aspect_ratio = total_h / total_w
        grid_height = max(1, int(grid_width * aspect_ratio * 0.5))

        scale_x = grid_width / total_w
        scale_y = grid_height / total_h

        print(f"\n--- ASCII Превью (Фон: {bg_w}x{bg_h} | Объект: {fg_w}x{fg_h} at ({pos_x}, {pos_y})) ---")

        for gy in range(grid_height):
            line = ""
            for gx in range(grid_width):
                # Проецируем точку сетки консоли в виртуальные координаты сцены
                px = min_x + int(gx / scale_x)
                py = min_y + int(gy / scale_y)

                is_bg = (0 <= px < bg_w) and (0 <= py < bg_h)
                is_fg = (pos_x <= px < pos_x + fg_w) and (pos_y <= py < pos_y + fg_h)

                if is_fg and not is_bg:
                    line += "!"  # Объект за пределами фона
                elif is_fg and is_bg:
                    line += "#"  # Объект внутри фона
                elif is_bg:
                    line += "."  # Чистый фон
                else:
                    line += " "  # Пустота за пределами и фона, и объекта

            print(line)

        has_overflow = (pos_x < 0 or pos_y < 0 or (pos_x + fg_w) > bg_w or (pos_y + fg_h) > bg_h)
        if has_overflow:
            print("[ВНИМАНИЕ] Объект частично или полностью выходит за пределы фона (!)")
        print("-" * grid_width + "\n")

    def _resize(self, fg: Image.Image, scale: float) -> Image.Image:
        new_w = (fg.width // 64) * 64
        new_h = (fg.height // 64) * 64

        # Вычисляем новый размер и подгоняем под кратность 64
        raw_w = ceil(fg.width * scale)
        raw_h = ceil(fg.height * scale)
        
        # Округляем до ближайшего числа, кратного 64
        new_w = max(64, (raw_w // 64) * 64)
        new_h = max(64, (raw_h // 64) * 64)

        return fg.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    def move_interactive(self, 
                            bg_size: tuple[int, int], 
                            fg_size: tuple[int, int], 
                            initial_pos: tuple[int, int]) -> tuple[int, int]:
            current_pos = initial_pos
            while True:
                self.preview_ascii_layout(bg_size, fg_size, current_pos)
                user_input = input("Введите новые координаты 'X Y' (или Нажмите Enter/q для подтверждения): ").strip()
                
                if not user_input or user_input.lower() in ['q', 'yes', 'y']:
                    print(f"Фиксируем координаты: {current_pos}")
                    break
                
                try:
                    x, y = map(int, user_input.split())
                    current_pos = (x, y)
                except ValueError:
                    print("[Ошибка] Введите два целых числа через пробел, например: 500 600")
                    
            return current_pos

    def _generate_mask(self,
            alpha_print: Image.Image,
            background: Image.Image = None,
            mode: str = "edge",
            outer_pad = None,
            inner_pad = None
        ) -> Image.Image:
            """
            Генерация маски инпейнта на основе альфа-канала с поддержкой 4 режимов:
            - full: полная перерисовка картинки (рекомендуется низкая сила инпейнта)
            - edge: кольцевая маска (только шов)
            - background_primary: полная маска объекта + контекст фона
            - figure_primary: маска фона ВОКРУГ объекта (сам объект вырезан из маски)
            """
            if outer_pad is None:
                outer_pad = max((self.mask_inflate // 5) * 4, 1)  
                # Минимальный паддинг для корректной работы морфологии
            if inner_pad is None:
                inner_pad = max(self.mask_inflate // 5, 1) 
                # Залезаем внутрь в пять раз меньше
            mask_np = np.array(alpha_print.convert("L"))
            _, binary = cv2.threshold(mask_np, 128, 255, cv2.THRESH_BINARY)

            kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_pad * 2 + 1, outer_pad * 2 + 1))
            kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_pad * 2 + 1, inner_pad * 2 + 1))
            dilated = cv2.dilate(binary, kernel_outer, iterations=1)
            eroded = cv2.erode(binary, kernel_inner, iterations=1)
            if mode == "full":
                if background is None:
                    raise ValueError("Для режима 'full' требуется передать фон (background) для определения размеров маски.")
                # Создаем абсолютно белое полотно (255) по размерам ФОНА
                final_mask = np.array(Image.new("L", background.size, 255))
            elif mode == "background_primary":
                # Вся область объекта плюс контекст фона (расширенная область)
                final_mask = dilated
            elif mode == "figure_primary":
                # Вся область картинки минус сам объект
                final_mask = cv2.bitwise_not(eroded)
            elif mode == "edge":
                #расширенная область минус суженная область = кольцо вокруг объекта
                final_mask = cv2.subtract(dilated, eroded)
            else:
                raise ValueError(f"Неизвестный режим: {mode}. Допустимые: 'full', 'edge', 'background_primary', 'figure_primary'")

            # Размытие краев для мягкой адаптации инпейнтера
            blurred = cv2.GaussianBlur(final_mask, (31, 31), 0)
            return Image.fromarray(blurred)

    def compose(self,
        background: Image.Image,
        figure: Image.Image,
        position: tuple[int, int] = (0, 0),
        scale: float = 1.0,
        mode: str = "full",
        #interactive: bool = False
        ) -> CompositionResult:
        #Корректный метод композиции для SDXL Inpaint
        
        # 0. Приводим ФОН к кратности 64
        bg_w = (background.width // 64) * 64
        bg_h = (background.height // 64) * 64
        if background.size != (bg_w, bg_h):
            background = background.resize((bg_w, bg_h), Image.Resampling.LANCZOS)

        # 1. Ресайз объекта (с гарантией кратности 64)
        fg_resized = self._resize(figure, scale)

        # 2. Определение координат
        if self.interactive:
            position = self.move_interactive(background.size, fg_resized.size, position)
        elif self.verbose:
            self.preview_ascii_layout(background.size, fg_resized.size, position)

        # 3. Извлечение Альфа-канала объекта
        if fg_resized.mode != "RGBA":
            raise ValueError(
                f"Для создания маски инпейнта требуется изображение с альфа-каналом (mode='RGBA'). "
                f"Получен формат: '{fg_resized.mode}'."
            )
        alpha_channel = fg_resized.split()[3]

        # 4. Генерация локальной маски инпейнта
        fg_mask_patch = self._generate_mask(alpha_channel, mode=mode, background=background)

        # 5. Сборка коллажа и маски одинакового размера
        collage = background.copy().convert("RGB")
        
        # Создаем пустой холст маски (черный = ничего не меняем)
        full_mask = Image.new("L", background.size, 0)

        # Вклеиваем объект на коллаж
        collage.paste(fg_resized, position, alpha_channel)

        # Вклеиваем маску, ИСПОЛЬЗУЯ альфа-канал как прозрачность при наложении,
        # чтобы края маски корректно совмещались с объектом
        full_mask.paste(fg_mask_patch, position)

        if mode == "full":
            # Для режима "full" маска должна быть полностью белой (255) по всей области объекта
            full_mask = Image.new("L", background.size, 255)

        # Важно: гарантируем, что маска возвращается в RGB, если этого требует пайплайн
        return CompositionResult(
            collage=collage, 
            mask=full_mask.convert("RGB")
        )