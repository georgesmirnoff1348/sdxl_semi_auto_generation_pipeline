from abc import abstractmethod, ABC
from PIL import Image
from pathlib import Path
from configs import FactorInferenceParameters, FactorPrompts
import time
from functools import wraps
from typing import Optional

class FactorDiffusor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_image(
        self, 
        prompts: FactorPrompts,
        config: FactorInferenceParameters, 
        save_path: Optional[Path] = Path("output.png")
        ) -> Image.Image:
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc, tb):
        self.unload()

    @abstractmethod
    def unload(self) -> None:
        """Метод очистки памяти, обязательный для реализации каждым конкретным диффузором."""
        pass
    

class FactorInpainter(ABC):
    def __init__(self):
        return self

    @abstractmethod
    def inpaint_image(
        self,
        prompts: FactorPrompts, 
        config: FactorInferenceParameters, 
        image: Image.Image, 
        mask: Image.Image,
        save_path: Optional[Path] = "output.png"
        ) -> Image.Image:
        pass

    def __exit__(self):
        self.unload()

    @abstractmethod
    def unload(self) -> None:
        """Метод очистки памяти, обязательный для реализации каждым конкретным диффузором."""
        pass


class FactorCutter(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def remove_background(self, image: Image.Image, save_path: Optional[Path] = "output.png") -> Image.Image:
        pass


class FactorImageReader(ABC):
    @abstractmethod
    def read_image(self, path: Path) -> Image.Image:
        pass


def factortimeinference(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Если декорируется метод класса, извлекаем имя класса из args[0]
        class_name = args[0].__class__.__name__ if args else ""
        algo_name = f"{class_name}.{func.__name__}"
        
        print(f"--- СИСТЕМА Ф.А.К.Т.О.Р.: ЗАПУСК АЛГОРИТМА {algo_name} ---")
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        
        print(f"--- СИСТЕМА ФАКТОР: {algo_name} ОТРАБОТАЛА ЗА {execution_time:.4f} СЕК ---")
        print(f"--- СИСТЕМА ФАКТОР: ОТЧЕТ О ПОТРАЧЕННОМ ЭВМ-ВРЕМЕНИ НАПРАВЛЕН РУКОВОДСТВУ НИИ 112-10 ---")
        return result
    return wrapper

