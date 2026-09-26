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

<a name="datasetgen"></a>

# datasetgen

Компактная библиотека + CLI поверх существующего SDXL-пайплайна: один YAML →
детерминированный план выходов → датасет в `output_dir` с журналом попыток
(`manifest.jsonl`), resume-логикой и перегенерацией отдельных кадров.

`datasetgen` — **аддитивный слой оркестрации**: он вызывает существующие обёртки
(`OrdinaryGen`, `Cutter`/`U2netCutter`/`BirefNetCutter`, `BackInpainter`,
`configs.FactorPrompts`, `configs.FactorInferenceParameters`) и не строит
пайплайны сам, не переписывает и не отключает их логи (`@factortimeinference`).

**Вход пакета — YAML-конфиг, а не естественный язык.** Сценария «напишите задачу
словами» нет: `datasetgen` не понимает свободный текст и не вызывает агентов.
Файл `dataset_description_request.md` — это опциональная заготовка для
**ручного** копирования в любой внешний чат/агент (пишется только при
`dataset_description.enabled: true` и заданном `request_file`), а не вызов
сервиса. Внешних вызовов (subprocess, HTTP, CLI-агенты) в пакете нет.

Пакет работает в двух режимах, и документация ниже разделена именно по ним:

* **из клона репозитория** — через `uv` (раздел 2.1);
* **установленным колесом через `pip`** — из любого окружения, исходный checkout
  не нужен (раздел 2.2).

## 1. Prerequisites

* **Python `>=3.12`** (`requires-python` в [`pyproject.toml`](pyproject.toml)).
  Это ограничение задаёт не код, а ML-стек: сам код держится на 3.9
  (`from __future__ import annotations`, `Path.is_relative_to()`), но
  **numpy 2.5.2 требует `>=3.12`**. Значения 3.11 недостаточно — понизить
  `requires-python`, не сломав extras, нельзя.
* **[uv](https://docs.astral.sh/uv/)** — обязателен для разработки и запуска из
  клона: зафиксированный `uv.lock`, `uv sync --locked`, `uv run --locked`,
  сборка колеса. Для **использования** uv не нужен — достаточно `pip` и
  готового колеса.
* **GPU (CUDA)** — только для реальной генерации: целевые рантаймы
  Kaggle/Colab. Apple Silicon (MPS) поддерживается legacy-обёртками.
* **сеть** — только для первого `uv sync` / `uv build` (пакеты и build-backend
  `hatchling`), первого `pip install` с extras и первого скачивания весов модели
  с Hugging Face;
* **`onnxruntime`** — нужен для `generation.mode: two_layer`. Он объявлен в
  extra `two-layer` **явно**, потому что `rembg 2.0.81` его транзитивно не
  тянет, а без него `rembg` **завершает процесс** (не найден ни один execution
  provider). Ставится CPU-сборка (`onnxruntime` совместима с macOS arm64); на
  CUDA-машинах при необходимости заменяется вручную на `onnxruntime-gpu` — такая
  сборка существует только под Linux/Windows.

### Имена: дистрибутив, пакет и команда

| Что | Имя |
|---|---|
| дистрибутив (pip, файл колеса) | `ai-basic-alina`, версия `0.1.0` |
| пакет для импорта | `datasetgen` |
| консольная команда | `datasetgen` (`datasetgen = "datasetgen.cli:main"`) |

Расхождение `ai-basic-alina` / `datasetgen` намеренное: дистрибутив назван по
репозиторию, команда и импорт — по назначению. В сообщениях об ошибках всегда
печатается `datasetgen`, а не `ai-basic-alina`.

Локально, без видеокарты, доступны `--help`, `--version` и `--dry-run`:
**`--dry-run does not construct pipelines, download models, generate images, or require a GPU`.**

## 2. Install

### 2.1 Разработка и запуск из клона

```bash
uv sync --locked
uv run --locked datasetgen --version     # datasetgen 0.1.0
```

Точка входа регистрируется в `pyproject.toml`:
`datasetgen = "datasetgen.cli:main"`, сборка — `hatchling`. Группа `dev` — это
`ai-basic-alina[two-layer]` плюс `scikit-image` и `torchvision`, то есть в
окружении разработки сразу есть всё, что нужно и `single_layer`, и `two_layer`,
и тестам. Первый `uv sync` собирает проект, поэтому доступ к сети нужен один раз.

### 2.2 Установка колеса через pip

```bash
uv build --wheel --out-dir dist
```

`uv build` сам печатает имя собранного колеса — подставьте его вместо
`<wheel-file>`: имя зависит от версии, платформы и имени дистрибутива, поэтому в
README оно и не зашито.

```bash
pip install 'dist/<wheel-file>'                 # help / version / dry-run / планирование
pip install 'dist/<wheel-file>[single-layer]'   # реальная генерация single_layer
pip install 'dist/<wheel-file>[two-layer]'      # + two_layer (rembg + onnxruntime)
```

* **базовая установка лёгкая**: единственная зависимость — `pyyaml`. Работают
  `--help`, `--version`, `--dry-run`, планирование, манифест и отчёты
  (`dataset_description.md`). `torch`, `diffusers`, `rembg` не нужны; реальная
  генерация без них честно падает с кодом **9** (раздел 13), а не «падает как-то
  внутри пайплайна»;
* extra **`single-layer`** — `accelerate`, `diffusers`, `numpy`,
  `opencv-python>=5.0.0.93`, `pillow`, `safetensors`, `torch`, `transformers`:
  нужен для `generation.mode: single_layer`;
* extra **`two-layer`** — это `single-layer` + `onnxruntime>=1.17.0` +
  `rembg>=2.0.81`: нужен для `generation.mode: two_layer`;
* колесо **самодостаточно**: после `pip install` исходный checkout не требуется
  ни для запуска, ни для примеров, ни для legacy-модулей.

#### Что входит в колесо (28 записей)

* пакет `datasetgen/`;
* пять legacy-модулей как **top-level** модули: `configs.py`, `cutter.py`,
  `cuda_mps_gens.py`, `diffusors_core.py`, `filesystems_core.py`;
* два примера задач как `datasetgen/examples/classification.yaml` и
  `datasetgen/examples/anomaly.yaml`.

`main.py` в колесо **не входит** (ручной сценарий репозитория), как и `tests/`,
`archive/`, `gen/`, `old sh/`.

#### Known limitation: коллизия top-level имён

Legacy-модули устанавливаются в `site-packages` как top-level, а имена `configs`
и `cutter` достаточно общие. Если в том же окружении окажется чужой пакет с
модулем `configs`, Python импортирует **чужой** файл: наш каталог добавляется в
**конец** `sys.path`, а не в начало — это требование идемпотентности shim
(`ensure_legacy_importable()`), который не должен ломать импорты в
`site-packages`.

Диагностика — одна команда:

```bash
python -c "import configs; print(configs.__file__)"
```

Если в выводе не `.../site-packages/configs.py` из нашего колеса — в окружении
конфликт имён, и `single_layer`/`two_layer` упадут невнятно. Лечится venv без
конфликтующего пакета.

Если модуль не импортируется (например, в базовой установке нет `torch`, который
`configs.py` тянет на верхнем уровне), проверьте путь через `find_spec` — он
ничего не выполняет:

```bash
python -c "import importlib.util as u; print(u.find_spec('configs').origin)"
```

#### Где взять пример YAML у установленного пакета

В checkout это [`examples/classification.yaml`](examples/classification.yaml) и
[`examples/anomaly.yaml`](examples/anomaly.yaml). В установленном пакете те же
файлы лежат в `<site-packages>/datasetgen/examples/`. Универсальный способ узнать
путь (работает в обоих режимах):

```bash
EX=$(python -c "from datasetgen import paths; print(paths.examples_dir())")
datasetgen run "$EX/classification.yaml" --model-id stabilityai/stable-diffusion-xl-base-1.0 --dry-run
```

### 2.3 Проверка установки wheel

```bash
uv run --locked python tools/verify_wheel.py
```

Скрипт [`tools/verify_wheel.py`](tools/verify_wheel.py) (только стандартная
библиотека) собирает колесо, проверяет его содержимое через `zipfile`, создаёт
временный **чистый venv** и ставит туда wheel **без исходного репозитория**
(`PYTHONPATH` очищен, `VIRTUAL_ENV` удалён, `PYTHONNOUSERSITE=1`, cwd вне
репозитория), после чего проверяет:

* `--help`, `--version` и `--dry-run` по примеру из колеса;
* разрешимость legacy-модулей из `site-packages` через `find_spec`;
* лёгкость базовой установки: `torch`, `diffusers`, `rembg`, `onnxruntime` → `None`;
* preflight режима `two_layer` → код выхода 9.

Итого **79 проверок**; при успехе скрипт печатает `OK: wheel verified`.

Та же проверка есть автотестом —
[`tests/test_wheel_install.py`](tests/test_wheel_install.py); по умолчанию он
**skip**, включается переменной окружения:

```bash
DATASETGEN_WHEEL_E2E=1 uv run --locked python -m unittest tests.test_wheel_install -v
```

Опции скрипта: `--keep-temporary` (не удалять временный каталог), `--python`
(версия интерпретатора для чистого venv, по умолчанию `3.13`), `--source-root`
(корень исходников для сборки, по умолчанию корень репозитория), `--timeout`
(таймаут одной внешней команды, по умолчанию 1800 с).

Требуются `uv` на `PATH` и сеть к PyPI: build-backend зафиксирован в
[`pyproject.toml`](pyproject.toml) как `hatchling==1.32.4` (в `uv.lock` он не
попадает, потому что `[build-system]` в lockfile не фиксируется), поэтому первая
изолированная сборка один раз скачивает ровно эту версию с PyPI и кладёт её в
кэш `uv`. Модели не скачиваются, пайплайны не строятся, GPU не нужен.

**Что это предотвращает** — типичную поломку «работает из checkout, но падает
после `pip install`»: забытый legacy-модуль в `force-include`, потерянный
`examples/` в колесе, неверная точка входа, случайно уехавший в колесо
`main.py`, подтянувшиеся в базовую установку тяжёлые зависимости, сломанный
preflight. Всё это ловится без единого реального GPU-прогона.

### 2.4 Автоматическая проверка в CI

Workflow [`.github/workflows/wheel.yml`](.github/workflows/wheel.yml) запускается
на каждом `push` и `pull_request` и выполняет ровно три команды:

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -t . -v
uv run --locked python tools/verify_wheel.py
```

* Python ставится по файлу [`.python-version`](.python-version) (`3.13`,
  `setup-python` с `python-version-file`), uv — фиксированной версии через
  `astral-sh/setup-uv`, кэш ключуется по `uv.lock`;
* любой ненулевой код возврата роняет job: ошибки сети, сборки и тестов **не**
  маскируются, «зелёный» статус получается только честным прогоном всех трёх
  шагов;
* набор проверок ровно тот же, что и локально, поэтому воспроизведение CI
  сводится к этим трём командам;
* реальная генерация изображений в CI **не** выполняется: веса моделей не
  качаются, GPU не нужен (см. «Что не проверено локально»).

## 3. CLI discovery

```bash
uv run --locked datasetgen --help
uv run --locked datasetgen --version     # datasetgen 0.1.0
```

В epilog `--help` есть блок `extras:` с готовыми командами установки
(`pip install 'ai-basic-alina[single-layer]'`, `pip install
'ai-basic-alina[two-layer]'` и вариант для локального колеса) и полная таблица
кодов выхода, включая `9 DEPENDENCY_MISSING`.

Установленная копия печатает тот же help: содержимое `--help` не зависит от
способа установки.

## 4. Полный YAML-пример, поле за полем

Пример задачи: [`examples/classification.yaml`](examples/classification.yaml)
(в установленном пакете — `datasetgen/examples/classification.yaml`, путь
узнаётся через `paths.examples_dir()`, см. 2.2):

```yaml
version: 1                             # int, REQUIRED, ровно 1
task: classification                   # enum, REQUIRED: classification | anomaly_detection
output_dir: ./out/classification      # str, REQUIRED; относительно CWD; не может быть существующим файлом
seed: 20260926                         # int, REQUIRED, >= 0; корневой сид всей работы

dataset_description:                   # OPTIONAL (по умолчанию выключено)
  enabled: true                        # bool, default false; true -> в конце прогона пишется dataset_description.md
  request_file: null                   # str|null, default null; НЕ выполняется, это путь к пользовательскому брифу
  max_prompt_rows: 50                  # int>0, default 50; сколько строк таблицы промптов печатать

generation:                            # REQUIRED
  mode: single_layer                   # enum, REQUIRED: single_layer | two_layer
  cutter: u2net                        # enum u2net|birefnet|custom, default u2net; только для two_layer
  cutter_model_name: null              # str, default null; ОБЯЗАТЕЛЕН при cutter: custom
  background:                          # REQUIRED IFF mode == two_layer
    model_id: diffusers/stable-diffusion-xl-1.0-inpainting-0.1   # str, REQUIRED (модель инпейнтинга)
    template: "background photography of {place}, 1980s, casual photo"   # str, REQUIRED
    negative_prompt: "people, illustration, human, outdoor, text"       # str, default ""
    variables:                         # REQUIRED, непустой mapping
      place: [hospital room with painted wall, industrial hall]
    inference:                         # OPTIONAL, значения по умолчанию как у inference
      num_inference_steps: 20           # int>0
      guidance_scale: 7.0               # float>=0
      width: 1024                       # int, кратно 8, 64..2048
      height: 1024                      # int, кратно 8, 64..2048
      extra:                            # -> FactorInferenceParameters.extra (для BackInpainter: inner_pad, strength)
        inner_pad: 20
        strength: 0.9

inference:                             # OPTIONAL (дефолты совпадают с FactorInferenceParameters)
  num_inference_steps: 20              # int>0, default 20
  guidance_scale: 7.0                  # float>=0, default 7.0
  width: 1024                          # int, default 1024, кратно 8, 64..2048
  height: 1024                         # int, default 1024, кратно 8, 64..2048
  extra: {}                            # mapping, default {} — произвольные kwargs пайплайна

classes:                               # REQUIRED, список; порядок = порядок обработки
  - name: class_a                      # str, REQUIRED; ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$; не зарезервировано; уникально
    count: 12                          # int>=1, REQUIRED
    template: "a {worker} wearing {uniform}, {pose}, studio photography"  # str, REQUIRED; {name} -> ^[a-z][a-z0-9_]{0,31}$
    negative_prompt: "3d render, anime, blurry, text, watermark"           # str, default ""
    variables:                         # REQUIRED, непустой mapping; каждый ключ обязан встречаться в шаблоне и наоборот
      worker: [factory worker, industrial engineer]
      uniform: [complete protective equipment, helmet and high-visibility workwear]
      pose: [standing naturally, inspecting machinery]
  - name: class_b
    count: 12
    template: "a {worker} wearing {uniform}, {anomaly}, concrete background, 1980s photo"
    negative_prompt: "hat, green, smooth skin, 3d render, anime, cartoon"
    variables:
      worker: [factory worker, industrial engineer]
      uniform: [complete protective equipment, helmet and high-visibility workwear]
      anomaly:
        - "thick rough grey concrete crust growing on the face skin"
        - "cracked grey cement scaling on the skin, grotesque appearance"
```

Правила, которые ловит валидация (все ошибки — код выхода **4 CONFIG**):

* `classification` → ровно 2 класса; `anomaly_detection` → ровно 1 класс с именем `normal`;
* неизвестный ключ в корне или в любом известном блоке → `unknown key in <блок>: '<key>' (allowed: ...)`;
* `bool` не принимается как `int` (`version`, `count`, `seed`, `num_inference_steps`);
* `{{` / `}}` в шаблоне — экранирование и дают литеральные `{` / `}`;
* `single_layer` + блок `background` — не ошибка, а предупреждение `unused two_layer block`;
* `request_file` должен быть существующим читаемым файлом: если он недоступен,
  `datasetgen` печатает `datasetgen: warning: ...` и всё равно создаёт `dataset_description.md`.

Валидация не зависит от установленных extras: даже «голый» Python с одним
`pyyaml` отличает корректный YAML от мусора. Проверка того, что режиму
*генерации* хватает пакетов, — отдельный preflight (код 9, раздел 13).

[`examples/anomaly.yaml`](examples/anomaly.yaml) — тот же формат, но
`task: anomaly_detection`, единственный класс `normal` и
`output_dir: ./out/anomaly`.

## 5. Безопасный локальный план (dry-run)

Из клона:

```bash
uv run --locked datasetgen run examples/classification.yaml \
    --model-id stabilityai/stable-diffusion-xl-base-1.0 --dry-run
```

```bash
uv run --locked datasetgen run examples/anomaly.yaml \
    --model-id stabilityai/stable-diffusion-xl-base-1.0 --dry-run
```

Из установленного пакета (путь к примеру берётся из самого пакета, см. 2.2):

```bash
EX=$(python -c "from datasetgen import paths; print(paths.examples_dir())")
datasetgen run "$EX/classification.yaml" --model-id stabilityai/stable-diffusion-xl-base-1.0 --dry-run
```

`--dry-run` **не конструирует пайплайны, не скачивает модели, не генерирует
изображения и не требует GPU**: он печатает ровно тот же план, который выполнит
реальный `run`, и не создаёт ни каталогов, ни файлов, ни `manifest.jsonl`
(`dry-run: nothing was written ...`). Поэтому `--dry-run` работает и на базовой
установке без extras: preflight зависимостей при нём не выполняется.

## 6. Реальная генерация (GPU: Kaggle/Colab)

```bash
uv run --locked datasetgen run examples/classification.yaml \
    --model-id stabilityai/stable-diffusion-xl-base-1.0
```

Для установленного колеса — та же команда без `uv run --locked` и с `$EX/...`
вместо `examples/...`.

* `--model-id` не пишется в YAML: он уходит в `OrdinaryGen(model=...)` и в манифест;
* в `two_layer` инпейнтер использует свой `generation.background.model_id`;
* первый запуск скачивает веса модели (сеть нужна только на первое скачивание);
* до первой записи на диск срабатывает preflight: если режиму не хватает пакетов —
  код **9** с точным списком отсутствующих модулей (раздел 13);
* прогон печатает построчный прогресс в stdout и итоговую строку
  `summary: planned=… succeeded=… failed=… already_complete=… rejected_moved=…`;
* код возврата: `0` — всё хорошо (включая «всё уже готово»), `7` — есть `failed`
  (батч при этом доходит до конца: одна плохая картинка не убивает прогон);
* если делать нечего (`planned = 0`), ни один пайплайн не конструируется —
  это честный no-op, который не падает даже на машине без CUDA/MPS.

**Про `two_layer` и ресурсы — без приукрашивания.** В текущей реализации режим
освобождает ресурсы **между кадрами**: в
[`Runner._two_layer_generator()`](datasetgen/runner.py) `OrdinaryGen`, `Cutter` и
`BackInpainter` создаются заново для каждого выхода и выгружаются сразу после
своего кадра, поэтому генератор и инпейнтер поднимаются заново на каждый кадр.
`single_layer` так не делает — там генератор создаётся лениво и живёт весь прогон
(см. `provider()` в [`datasetgen/runner.py`](datasetgen/runner.py)). Отсюда
следствие: `two_layer` экономнее по VRAM (ниже пик памяти, между кадрами не
ничего не держится), но **значительно медленнее** `single_layer`. Это сознательный
компромисс в пользу предсказуемой памяти, а **не** доказательство высокой
производительности: тайминги и расход VRAM на настоящем GPU не измерялись (см.
«Что не проверено локально» в конце документации).

**Этот путь локально не прогонялся**: он требует GPU, CUDA/MPS и скачивания
весов. Что именно проверено, а что нет, — в конце документации.

## 7. Resume

```bash
uv run --locked datasetgen run examples/classification.yaml \
    --model-id stabilityai/stable-diffusion-xl-base-1.0
```

Повтор той же команды безопасен: для каждого `(класс, индекс)` кадр считается
**готовым**, только если последняя запись в манифесте — `succeeded` И файл
`output_dir/<класс>/<класс>_<index>.png` существует и не пуст. Всё остальное
(нет записей, только `planned`, только `failed`, `succeeded` без файла) —
незавершённая работа: она дозаполняется, готовые файлы не перезаписываются,
не перемещаются и не удаляются. Незавершённая `planned`-попытка продолжается с
тем же номером `attempt`, поэтому её промпт и seed не меняются.

Если упавшая попытка оставила после себя файл (например, бэкенд записал кадр и
упал при выгрузке модели), этот файл описан в манифесте и считается своим:
он **перезаписывается** новой попыткой с `attempt + 1`, и прогон не падает с
`UNSAFE_PATH`. `UNSAFE_PATH` возникает только для файла, о котором манифест не
знает ни одной записи (см. код 6 в разделе 13).

## 8. Отклонение кадров и перегенерация

```bash
uv run --locked datasetgen regenerate examples/classification.yaml \
    --model-id stabilityai/stable-diffusion-xl-base-1.0 \
    --remove "class_a:3,7-9;class_b:4"
```

```bash
uv run --locked datasetgen regenerate examples/anomaly.yaml \
    --model-id stabilityai/stable-diffusion-xl-base-1.0 \
    --remove "3,7-9"
```

```bash
uv run --locked datasetgen regenerate examples/classification.yaml \
    --model-id stabilityai/stable-diffusion-xl-base-1.0 \
    --remove "class_a: 1-3, 5"
```

Грамматика `--remove`:

```
SELECTOR := SEGMENT (";" SEGMENT)*        # хотя бы один SEGMENT
SEGMENT  := [CLASS ":"] ITEM ("," ITEM)+  # хотя бы один ITEM
ITEM     := INT | INT "-" INT              # включительно
INT      := [0-9]+                          # >= 1
```

`"class_a:3,7-9;class_b:4"` → `class_a: 3, 7, 8, 9` и `class_b: 4`.
Для `anomaly_detection` класс можно опустить — он применится к `normal`.
Ошибки селектора — код выхода **5 SELECTOR**: `0`, `9-7`, `class_a:25` (вне
диапазона), `class_z:3`, повторы/пересечения (`duplicate index …`) и т. д.

Что происходит с отклонённым кадром:

1. существующий `class_a/class_a_3.png` **перемещается** в
   `out/_rejected/class_a_3_1.png` (тот же файл, те же байты, ничего не удаляется);
2. в манифест дописывается запись `rejected` для старой попытки
   (`rejected_file`, `reason: "cli --remove"`);
3. планируется новая попытка `attempt = 2`: **новый `seed` и новая реализация
   промпта** (номер попытки входит в формулу детерминизма);
4. под тем же номером `class_a_3.png` создаётся новое изображение;
5. если новая пара `(seed, prompt)` совпала бы с уже записанной — выход
   помечается `failed` с `error.stage = "guard"` и сообщением
   `refusing to reproduce an identical (seed, prompt) realization …`, файл не трогается.

Если переносить нечего, запись `rejected` всё равно пишется и документирует
отказ: `rejected_file: null`, `reason: "cli --remove"`, `rejected_at`. Файл
при этом не трогается, `attempt` всё равно увеличивается.

Если предыдущая попытка завершилась статусом `failed` (готового кадра она не
дала), запись `rejected` НЕ создаётся: в графе статусов нет перехода
`failed -> rejected`. Выводится предупреждение о замене остатка, и новый кадр
записывается под тем же номером.

## 9. Что получается в `output_dir`

```
out/classification/
├── class_a/                       class_a_1.png … class_a_12.png
├── class_b/                       class_b_1.png … class_b_12.png
├── _rejected/                     class_a_3_1.png  (отклонённые, перемещены, не удалены)
├── _intermediate/                 создаётся ТОЛЬКО при mode: two_layer
│   └── class_a/                   class_a_1_1_subject.png, class_a_1_1_cut.png
├── manifest.jsonl                 append-only журнал попыток
├── dataset_description.md         отчёт (dataset_description.enabled: true)
└── dataset_description_request.md заготовка для внешнего агента
                                   (enabled: true И задан request_file)
```

Для `anomaly_detection` раскладка та же, но единственный каталог — `normal/`
(`normal_1.png`, …). Ничего, кроме перечисленного, в корень `output_dir`
раннер не пишет.

Куда нельзя писать, зависит от режима ([`datasetgen/paths.py`](datasetgen/paths.py);
флаг `IS_SOURCE_CHECKOUT` в [`datasetgen/bootstrap.py`](datasetgen/bootstrap.py)
определяется по наличию `pyproject.toml` рядом с пакетом):

* **из клона** защищены `archive/`, `gen/`, `old sh/`, `examples/` и сам каталог
  пакета `datasetgen/`, плюс действует правило «`output_dir` не может быть
  предком корня репозитория». Всё это — код **6 UNSAFE_PATH**;
* **установленный пакет** защищён только каталогом `datasetgen/`. Правило про
  «предка корня репозитория» здесь **не действует намеренно**: корня
  репозитория нет, `repo` указывает на `site-packages`, и сравнение с ним
  отвергало бы `~/.local`, `/usr/local`, `$HOME` и самый частый случай — каталог
  проекта, внутри которого создан venv (примеры кладут вывод ровно в `./out/...`).

Базовые проверки безопасности работают в обоих режимах: `output_dir` не может
быть существующим файлом, не может содержать job-файл, не может быть непустым
каталогом без `manifest.jsonl`, а все пути строятся через `safe_join` (выход за
пределы `output_dir` — тот же код 6).

## 10. Манифест

`manifest.jsonl` — UTF-8 JSONL (`ensure_ascii=false`, `sort_keys=True`),
только дописывается; перезаписей нет. Два типа записей, у каждой
`schema_version: 1`; пути — POSIX-**относительные** от `output_dir`:

* `run` — одна на запуск (`command`, `job_sha256`, `output_dir`, `task`,
  `generation_mode`, `model_id`, `job_seed`, `classes`, `inference`, `two_layer`,
  а для `regenerate` — `removed`);
* `image` — по одной на попытку (`class`, `index`, `attempt`, `status`,
  `output`, `prompt`, `negative_prompt`, `seed`, `model_id`, `settings`,
  `intermediate`, `timestamp`, `error`, `bytes`, `duration_sec`).

Переходы статусов:

```
(нет)         --planned-->  planned
planned       --успех----->  succeeded
planned       --ошибка---->  failed
succeeded     --regenerate-> rejected   (файл перемещён в _rejected/)
failed        --run/regen-> planned (attempt+1, новый seed и новая реализация)
rejected      --regenerate-> planned (attempt+1)
```

Перехода `failed -> rejected` нет: у failed-попытки нечего отклонять. Запись
`rejected` всегда содержит `rejected_file` (строка или `null`), `reason` и
`rejected_at`.

Эффективное состояние `(class, index)` = последняя `image`-запись в файле.
Пустой/отсутствующий манифест — чистый старт. Битые строки пропускаются с
`datasetgen: warning: manifest: skipping corrupt line N`; если файл непуст,
но валидных записей нет — код **8 MANIFEST**. Незавершённый хвост (строка без
финального `\n`) усекается до последней целой записи. Изменение самого YAML
между прогонами даёт предупреждение `job file changed since the last run (…)`,
а рост `count` полностью легален (генерируются только новые индексы); при
уменьшении `count` лишние файлы остаются на диске.

### Воспроизводимость

* повтор той же попытки даёт тот же `prompt` и тот же `seed` (формула `datasetgen/v1`);
* байт-в-байт воспроизводимость пикселей **не гарантируется**: revision
  репозитория модели не закреплена, а результат зависит от fp16, attention
  implementation, версий CUDA/драйвера, CUDA vs MPS и версий torch/diffusers;
* отклонённая попытка намеренно никогда не воспроизводится: `attempt` входит в
  material, поэтому seed новой попытки отличается, а guard сверяет новую пару
  `(seed, prompt)` со **всеми** прошлыми реализациями этого индекса (а не только с
  последней) и при совпадении помечает попытку `failed` с `error.stage = "guard"`,
  не перезаписывая и не перемещая файл;
* известное ограничение legacy-кода: `OrdinaryGen.__init__` / `ControlNetGen.__init__`
  не проставляют `self.device` без CUDA/MPS и падают с `AttributeError` —
  реальная генерация рассчитана на GPU (Kaggle/Colab).

## 11. Файлы dataset-description

Ещё раз, чтобы не было двусмысленности: пакет принимает **YAML**, а не
естественный язык. Файлы ниже — обычные локальные отчёты, которые пользователь
сам решает, куда девать.

`dataset_description.enabled: false` (по умолчанию) — файлы не создаются.

`enabled: true` — в конце `run`/`regenerate` атомарно создаются:

* **`dataset_description.md`** — только факты из конфига и манифеста: задача и
  раскладка, таблица классов (запрошено/произведено/отклонено), источник
  промптов (шаблоны и словари переменных, включая фон), таблица реализаций
  `класс | index | attempt | seed | prompt` (для `two_layer` — отдельная колонка с
  фоновым промптом, усечение до `max_prompt_rows` с явной пометкой), параметры
  генерации, результаты (successes/failures/отклонённые/сообщения об ошибках),
  раздел воспроизводимости и происхождение текста: «Сгенерировано локально
  командой `datasetgen`; внешние агенты не вызывались.»;
* **`dataset_description_request.md`** — только если `enabled: true` И задан
  `request_file`: копия фактического отчёта, блок «ЗАДАЧА ДЛЯ ВНЕШНЕГО АГЕНТА» и
  исходный материал пользователя (файл только читается и не изменяется).
  Преамбула прямо говорит: «`datasetgen` не выполнял и не будет выполнять
  внешние команды; этот файл подготовлен для ручного копирования в любой
  чат-агент». Это ручная заготовка для копирования, а не интеграция с агентом.
  Если `request_file` недоступен — `datasetgen: warning: …`, а
  `dataset_description.md` всё равно создаётся.

Формулировки про происхождение и преамбула заданы константами в
[`datasetgen/description.py`](datasetgen/description.py) (`LOCAL_PROVENANCE`,
`AGENT_PREAMBLE`, `AGENT_TASK_BLOCK`); на их точность есть отдельные тесты.

Внешние вызовы (subprocess, HTTP, CLI-агенты) в пакете отсутствуют: это
проверяется тестами по исходникам.

## 12. Тесты

Тесты не обращаются к сети и не строят модели (фейки пишут 1×1 PNG в `tmp`).
Фактическое число — **228 тестов**:

```bash
uv run --locked python -m unittest discover -s tests -t . -v
```

```
Ran 228 tests in 6.0s

OK (skipped=1)
```

* лёгкий прогон (по умолчанию): **227 проходят**, 1 пропущен — wheel e2e
  (раздел 2.3);
* полный прогон с `DATASETGEN_WHEEL_E2E=1`: **228 passed, 0 skipped** (нужна
  сеть к PyPI).

Отдельно проверяется, что тяжёлые модули не импортируются ни при импорте CLI,
ни при `--dry-run`, и что preflight зависимостей честно падает с кодом 9, а не
молча тянет тяжёлое.

Из hardening-проверок (обе без сети и без установленного колеса):

* [`tests/test_paths.py`](tests/test_paths.py) — поведение **после установки**:
  при `IS_SOURCE_CHECKOUT = False` каталог `site-packages` не считается корнем
  исходного репозитория, но сам installation/package directory и его предки
  запрещены к записи; `examples_dir()` в установленном режиме отдаёт
  `datasetgen/examples` (package data), а в checkout — `examples/` репозитория.
  Режим подменяется фейковым `site-packages` во временном каталоге через
  `mock.patch`, реально установленный wheel не нужен;
* [`tests/test_dependencies.py`](tests/test_dependencies.py) — соответствие
  extras `single-layer` / `two-layer` из [`pyproject.toml`](pyproject.toml) списку
  модулей, которые проверяет preflight, вместе с известными несовпадениями имён
  `pillow → PIL` и `opencv-python → cv2` (раздел 13).

## 13. Коды выхода

| Код | Имя | Когда |
|---|---|---|
| 0 | OK | успех, `--dry-run`, «всё уже готово» (no-op, пайплайн не строится) |
| 1 | INTERNAL | непойманное исключение (+ traceback) |
| 2 | USAGE | ошибки argparse |
| 3 | JOB_NOT_FOUND | YAML не найден или это не файл |
| 4 | CONFIG | ошибка валидации YAML |
| 5 | SELECTOR | ошибка `--remove` |
| 6 | UNSAFE_PATH | опасный `output_dir`; чужой файл без записи в манифесте; непустой `output_dir` без `manifest.jsonl`; `output_dir` расходится с манифестом; занятый путь в `_rejected/` |
| 7 | GENERATION_FAILED | есть записи `failed` |
| 8 | MANIFEST | манифест нечитаем/бит и не дал валидных записей |
| 9 | DEPENDENCY_MISSING | выбранному `generation.mode` не хватает пакетов |

Ошибки печатаются в stderr как `datasetgen: error: [CODE] <сообщение>`,
предупреждения — как `datasetgen: warning: <текст>` и код возврата не меняют.

### Код 9: DEPENDENCY_MISSING

```
datasetgen: error: [DEPENDENCY_MISSING] generation.mode: two_layer requires modules that are not installed: PIL, accelerate, cv2, diffusers, numpy, onnxruntime, rembg, safetensors, torch, transformers (install the extra: pip install 'ai-basic-alina[two-layer]')
```

Это дословный вывод в чистой базовой установке колеса (единственная зависимость —
`pyyaml`). Перечисляются **все** отсутствующие модули режима, а не только те, что
«пришли в `two_layer`»: `PIL`/`torch`/`diffusers` и `onnxruntime` приходят одним
extra, поэтому набор имён зависит от того, что уже стоит в окружении.

Preflight ([`datasetgen/engines.py`](datasetgen/engines.py)) срабатывает **до**
любой записи на диск и до legacy-импортов, только при реальных бэкендах и **не**
при `--dry-run`. Проверка идёт через `importlib.util.find_spec`: модуль только
находится, но не импортируется, поэтому тяжёлые зависимости не подтягиваются, а
`--dry-run` остаётся лёгким. Списки проверяемых модулей совпадают с extras
`single-layer` / `two-layer` из [`pyproject.toml`](pyproject.toml), поэтому
подсказка «install the extra» всегда точна.

Совпадение **проверяется тестом**, а не договорённостью: имена модулей
preflight должны в точности совпадать с набором дистрибутивов из extras (с
рекурсивным разворотом `ai-basic-alina[single-layer]` внутри `two-layer`).
Известные несовпадения «имя дистрибутива ≠ имя модуля» перечислены явно и
хранятся в одном месте — `DISTRIBUTION_TO_IMPORT` в
[`tests/test_dependencies.py`](tests/test_dependencies.py): `pillow → PIL`,
`opencv-python → cv2`. Новая зависимость в extra, для которой нет ни записи в
этом словаре, ни имени в `RUNTIME_MODULES`, валит тест с понятным сообщением —
добавить runtime-зависимость и забыть про preflight нельзя.

## Что не проверено локально

Честное ограничение, чтобы документация не обещала лишнего.

**Реальная генерация изображений не запускалась.** Для неё нужны GPU,
CUDA/MPS и скачивание весов с huggingface.co. Проверены только:

* упаковка и содержимое колеса, установка в чистом venv **без исходников**
  (раздел 2.3);
* CLI: `--help`, `--version`, `--dry-run` по обоим примерам, из клона и из
  установленного колеса;
* планирование, resume, перегенерация, манифест и отчёты — на фейках (тесты
  пишут 1×1 PNG в `tmp`), а не на настоящих картинках;
* preflight зависимостей и коды выхода, включая 9;
* разрешимость legacy-модулей и отсутствие тяжёлых импортов.

**Не** проверено: качество и содержимое реальных изображений, работа
SDXL/VAE/ControlNet, `rembg`/u2net на настоящих картинках, тайминги и расход
VRAM на GPU, поведение на Kaggle/Colab. Всё, что описано в разделах 6, 8 и в
«известных ограничениях legacy-кода», — это документация ожидаемого поведения,
а не результат локального прогона.
