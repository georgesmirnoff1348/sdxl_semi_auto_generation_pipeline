from abc import abstractmethod, ABC
from PIL import Image
from pathlib import Path
from configs import FactorInferenceParameters, FactorPrompts
import time
from functools import wraps
from typing import Optional
import torch
import gc

class FactorDiffusor(ABC):
    @abstractmethod
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
        return self

    def __exit__(self, exc_type, exc, tb):
        self.unload()

    def unload(self) -> None:
        "You need to free your memory because diffusors are too heavy"
        print(
            "--- СИСТЕМА ФАКТОР: НАЧАТО ИЗВЛЕЧЕНИЕ МОДЕЛИ ИЗ ОПЕРАТИВНОЙ ПАМЯТИ ---"
        )

        if hasattr(self, "pipeline"):
            del self.pipeline

        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif self.device == "mps":
            torch.mps.empty_cache()

        print(
            "--- СИСТЕМА ФАКТОР: ОПЕРАТИВНАЯ ПАМЯТЬ УСПЕШНО ОСВОБОЖДЕНА ---"
        )
    

class FactorInpainter(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def __enter__(self):
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

    def __exit__(self, exc_type, exc, tb):
            self.unload()

    def unload(self) -> None:
            "You need to free your memory because diffusors are too heavy"
            print(
                "--- СИСТЕМА ФАКТОР: НАЧАТО ИЗВЛЕЧЕНИЕ МОДЕЛИ ИЗ ОПЕРАТИВНОЙ ПАМЯТИ ---"
            )
    
            if hasattr(self, "pipeline"):
                del self.pipeline
    
            gc.collect()
    
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            elif self.device == "mps":
                torch.mps.empty_cache()
    
            print(
                "--- СИСТЕМА ФАКТОР: ОПЕРАТИВНАЯ ПАМЯТЬ УСПЕШНО ОСВОБОЖДЕНА ---"
            )

    def unload(self) -> None:
        "You need to free your memory because diffusors are too heavy"
        print(
            "--- СИСТЕМА ФАКТОР: НАЧАТО ИЗВЛЕЧЕНИЕ МОДЕЛИ ИЗ ОПЕРАТИВНОЙ ПАМЯТИ ---"
        )

        if hasattr(self, "pipeline"):
            del self.pipeline

        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif self.device == "mps":
            torch.mps.empty_cache()

        print(
            "--- СИСТЕМА ФАКТОР: ОПЕРАТИВНАЯ ПАМЯТЬ УСПЕШНО ОСВОБОЖДЕНА ---"
        )
    

class FactorCutter(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def remove_background(self, image: Image.Image, save_path: Optional[Path] = "output.png") -> Image.Image:
        pass

"""
class FactorImageReader(ABC):
    @abstractmethod
    def read_image(self, path: Path) -> Image.Image:
        pass
"""

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

