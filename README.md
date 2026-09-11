[English](#english) | [Русский](#русский)

<a name="english"></a>

# Purpose of the Project
A lightweight utility pipeline built to generate synthetic dataset images for a custom computer vision project.

# diffusors_core.py
This module defines unified contracts for generation, inpainting, and segmentation models, while providing automatic memory cleanup for VRAM and RAM.

## Classes

* `FactorDiffusor(ABC)` — Base class for Text-to-Image / Image-to-Image models. Supports context management (with) and the unload() method for releasing pipelines from memory.
* `FactorInpainter(ABC)` — Base class for background inpainting and editing models (inpaint_image).
* `FactorCutter(ABC)` — Interface for background removal and segmentation (remove_background).

## Decorators

* `@factortimeinference` — Measures exact execution time (time.perf_counter) and outputs system logs structured as ClassName.method_name.

---

## Usage Example
```python
# Automatically unload the model from VRAM upon exiting the with block
with BackInpainter() as inpainter:
result = inpainter.inpaint_image(prompts, config, image, mask)
```

# filesystems_core.py
This module provides utilities for analyzing directory structures, searching for available numerical indices, and managing sequence patterns formatted as prefix_index.ext.

## Classes

### 1. DirectoryScanner
Base filesystem scanner.

* `count_files(recursive)` — Counts files in a target directory (with or without recursion).
* `find_by_prefix(prefix)` — Locates files matching a specific prefix.
* `analyze_pattern_files()` — Gathers full index statistics for prefix_index patterns (min/max indices, file counts, file lists).
* `get_next_available_filename(...)` — Finds the lowest available index to save new files without overwriting existing data.

### 2. ComradeManager(DirectoryScanner)
Specialized sequence manager tailored for indexed filenames (comrade_1.png, comrade_2.png, etc.).

* `find_missing_numbers()` — Detects missing index gaps within a sequence.
* `change_prefix(new_prefix)` — Safely renames prefixes across files while preserving original indices and extensions.
* `compress_and_renumber(start_from, new_prefix)` — Closes index gaps by performing a safe two-pass sequential renumbering.
* `get_next_filename(...) `— Provides fast lookup for the next free filename using the active prefix.

---

## Usage Example
```python
from dir_scanner import ComradeManager
# Compress indices and eliminate sequence gaps automatically
manager = ComradeManager("./output_images", prefix="frame")
# Identify missing frames (e.g., [3, 7])
missing = manager.find_missing_numbers()
# Renumber all files sequentially: frame_1.png, frame_2.png...
manager.compress_and_renumber(start_from=1)
```
# cutter.py
Provides class wrappers for automated object extraction and RGBA mask generation using the rembg library (ONNX Runtime engine).

## Classes

### 1. Cutter(FactorCutter)
Base segmentation class.

* Automated Provider Selection: Cascades hardware execution providers by availability: CUDAExecutionProvider -> MPSExecutionProvider -> CPUExecutionProvider.
* Core Method: remove_background(image, save_path) — Extracts the primary object and returns a transparent RGBA PIL image.

### 2. U2netCutter(Cutter)
Shortcut class for the standard u2net architecture. Lightweight and suitable for general-purpose segmentation.

### 3. BirefNetCutter(Cutter)
Shortcut class for high-resolution segmentation using birefnet-general. Delivers higher boundary precision on complex details such as hair, fine structures, or semi-transparent textures.

---

## Usage Example
```python
from cutter import BirefNetCutter
# Initialize BiRefNet for high-precision matting
cutter = BirefNetCutter()
# Extract object and receive RGBA output
character_rgba = cutter.remove_background(
image=studio_photo,
save_path=Path("outputs/character_no_bg.png")
)
```

# cuda_mps_gens.py
Contains generative pipeline implementations powered by Stable Diffusion XL (SDXL), with automatic hardware target detection for CUDA and Apple Silicon (MPS).

---

## Pipeline Classes

### 1. OrdinaryGen(FactorDiffusor)

Text-to-Image baseline generator using stabilityai/stable-diffusion-xl-base-1.0 (easily swappable with custom model weights).

* Key Features: Employs DPMSolverMultistepScheduler with Karras sigmas (use_karras_sigmas=True) alongside attention_slicing to lower peak VRAM consumption.
* Core Method: generate_image(prompts, config, save_path)

### 2. BackInpainter(FactorInpainter)
Dedicated background inpainting and outpainting pipeline designed to synthesize backgrounds around foreground objects.

* Key Features: Utilizes the lightweight VAE madebyollin/sdxl-vae-fp16-fix to eliminate fp16 numerical precision artifacts. Automatically adjusts foreground alpha masks via morphological erosion (configured via inner_pad) and inversion for seamless edge blending.
* Core Method: inpaint_image(prompts, config, image, mask, save_path)

### 3. ControlNetInpainter(FactorInpainter)
Guided inpainting powered by ControlNet architectures (defaults to xinsir/controlnet-tile-sdxl-1.0).

* Key Features: Enables guided background generation conditioned on reference imagery (control_image). Applies Gaussian blurring (GaussianBlur) across mask borders to soften transitions.
* Core Method: inpaint_ControlNet(prompts, config, image, alpha_print, control_image, save_path)

---

## Shared Capabilities
* Performance Tracking: All inference methods are wrapped in @factortimeinference for detailed time profiling.
* Reproducibility: Seed handling is standardized across devices via torch.Generator.
* Memory Management: Clears GPU caches (torch.cuda.empty_cache() / torch.mps.empty_cache()) automatically following generation runs.

---

## Usage Example
```python
from diffusors import BackInpainter
# Synthesize a background around an isolated character
with BackInpainter() as inpainter:
result = inpainter.inpaint_image(
prompts=prompts_data,
config=inference_config,
image=original_photo,
mask=character_alpha_mask,
save_path=Path("outputs/result.png")
)
```

<a name="русский"></a>

# Смысл существования проекта
Небольшой сборный пайплайн, который я делал для генерации синтетического датасета для своего проекта по машинному зрению. 

# diffusors_core.py
Модуль задает единые контракты для моделей генерации, инпаинтинга и сегментации, а также обеспечивает автоматическую очистку памяти (VRAM/RAM).

## Классы
* **`FactorDiffusor(ABC)`** — базовый класс для Text-to-Image / Image-to-Image моделей. Поддерживает контекстный менеджер (`with`) и метод `unload()` для выгрузки пайплайна из памяти.
* **`FactorInpainter(ABC)`** — базовый класс для моделей дорисовки и редактирования фона (`inpaint_image`).
* **`FactorCutter(ABC)`** — интерфейс для сегментации и удаления фона (`remove_background`).

## Декораторы
* **`@factortimeinference`** — замеряет точное время выполнения генерации (`time.perf_counter`) и выводит системные логи формата `ClassName.method_name`.

---

## Пример использования

```python
# Автоматическая выгрузка модели из VRAM при выходе из блока with
with BackInpainter() as inpainter:
    result = inpainter.inpaint_image(prompts, config, image, mask)
```

# filesystems_core.py

Модуль предоставляет утилиты для анализа директорий, автоматического поиска свободных индексов и управления файловыми последовательностями вида `префикс_№.ext`.

## Классы

### 1. `DirectoryScanner`
Базовый сканер файловой системы.
* **`count_files(recursive)`** — подсчет файлов в папке (с рекурсией или без).
* **`find_by_prefix(prefix)`** — поиск файлов по началу имени.
* **`analyze_pattern_files()`** — сбор полной статистики по шаблонам `префикс_№` (мин/макс номер, количество, список файлов).
* **`get_next_available_filename(...)`** — поиск наименьшего свободного порядкового номера для создания нового файла без перезаписи существующих.

### 2. `ComradeManager(DirectoryScanner)`
Специализированный менеджер для работы с файловыми последовательностями (`comrade_1.png`, `comrade_2.png` и т.д.).
* **`find_missing_numbers()`** — поиск "пропусков" (дыр) в нумерации файлов.
* **`change_prefix(new_prefix)`** — массовая смена префикса с сохранением оригинальных номеров и расширений.
* **`compress_and_renumber(start_from, new_prefix)`** — устранение пропусков в нумерации путем безопасного двухэтапного переименования по порядку.
* **`get_next_filename(...)`** — быстрый доступ к следующему свободному имени для текущего префикса.

---

## Пример использования

```python
from dir_scanner import ComradeManager

# Автоматическое сжатие нумерации и сброс пропусков
manager = ComradeManager("./output_images", prefix="frame")

# Находит пропущенные кадры (например, [3, 7])
missing = manager.find_missing_numbers() 

# Переименовывает все файлы по порядку: frame_1.png, frame_2.png...
manager.compress_and_renumber(start_from=1)
```

# cutter.py

Модуль предоставляет классы для автоматического вырезания объектов и создания RGBA-масок с использованием библиотеки `rembg` (ONNX Runtime).

## Классы

### 1. `Cutter(FactorCutter)`
Базовый класс для сегментации изображений.
* **Автовыбор провайдера:** Автоматически каскадирует аппаратное ускорение в порядке приоритета: `CUDAExecutionProvider` → `MPSExecutionProvider` → `CPUExecutionProvider`.
* **Основной метод:** `remove_background(image, save_path)` — вырезает объект и возвращает PIL-изображение с прозрачным фоном (RGBA).

### 2. `U2netCutter(Cutter)`
Ярлык (shortcut) для классической универсальной модели `u2net`. Быстро работает и подходит для большинства стандартных объектов.

### 3. `BirefNetCutter(Cutter)`
Ярлык для современной модели высокого разрешения `birefnet-general`. Обеспечивает более точную сегментацию сложных границ (волосы, мелкие детали, полупрозрачные ткани).

---

## Пример использования

```python
from cutter import BirefNetCutter

# Инициализация модели BiRefNet для точного вырезания
cutter = BirefNetCutter()

# Получение RGBA-изображения с вырезанным объектом
character_rgba = cutter.remove_background(
    image=studio_photo, 
    save_path=Path("outputs/character_no_bg.png")
)
```

# cuda_mps_gens.py
Модуль содержит реализации генеративных пайплайнов на базе **Stable Diffusion XL (SDXL)**, оптимизированных для работы с CUDA и Apple Silicon (MPS) (определяются автоматически).

---

## Классы пайплайнов

### 1. `OrdinaryGen(FactorDiffusor)`
Базовый генератор изображений по текстовому описанию (Text-to-Image) на базе `stabilityai/stable-diffusion-xl-base-1.0`. Возможно при желании заменить модель. 
* **Особенности:** Использует планировщик `DPMSolverMultistepScheduler` с сигмами Карраса (`use_karras_sigmas=True`) и `attention_slicing` для снижения потребления VRAM.
* **Основной метод:** `generate_image(prompts, config, save_path)`

### 2. `BackInpainter(FactorInpainter)`
Специализированный пайплайн для замены и генерации фона вокруг объектов (Inpainting / Outpainting).
* **Особенности:** Использует легковесный VAE `madebyollin/sdxl-vae-fp16-fix` для избежания артефактов fp16. Автоматически обрабатывает альфа-маску объекта: применяет морфологическую эрозию (контролируется параметром `inner_pad`) и инвертирует её для точной подгонки границ фона.
* **Основной метод:** `inpaint_image(prompts, config, image, mask, save_path)`

### 3. `ControlNetInpainter(FactorInpainter)`
Управляемый инпаинтинг с использованием сетей ControlNet (по умолчанию `xinsir/controlnet-tile-sdxl-1.0`).
* **Особенности:** Позволяет направлять генерацию фона с помощью опорного изображения (`control_image`). Автоматически сглаживает границы маски с помощью размытия по Гауссу (`GaussianBlur`).
* **Основной метод:** `inpaint_ControlNet(prompts, config, image, alpha_print, control_image, save_path)`

---

## Общие ключевые возможности

* **Замер времени:** Все методы генерации обернуты декоратором `@factortimeinference` для точного логирования времени работы.
* **Воспроизводимость:** Автоматическая обработка и фиксирование случайных зерен (`seed`) через `torch.Generator`.
* **Управление памятью:** Автоматическая очистка кэша GPU (`torch.cuda.empty_cache()` / `torch.mps.empty_cache()`) после каждого цикла генерации.

---

## Пример использования

```python
from diffusors import BackInpainter

# Генерация фона вокруг вырезанного персонажа
with BackInpainter() as inpainter:
    result = inpainter.inpaint_image(
        prompts=prompts_data,
        config=inference_config,
        image=original_photo,
        mask=character_alpha_mask,
        save_path=Path("outputs/result.png")
    )
```
