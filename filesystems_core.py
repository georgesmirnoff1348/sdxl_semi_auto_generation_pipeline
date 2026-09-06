import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class DirectoryScanner:
    """Класс для общих операций с файловой системой и анализа папки."""
    
    def __init__(self, directory_path: str | Path):
        self.dir_path = Path(directory_path)
        if not self.dir_path.exists() or not self.dir_path.is_dir():
            raise ValueError(f"Указанный путь {directory_path} не существует или не является директорией")

    def count_files(self, recursive: bool = False) -> int:
        #Подсчитать количество файлов в папке
        pattern = "**/*" if recursive else "*"
        return sum(1 for p in self.dir_path.glob(pattern) if p.is_file())

    def find_by_prefix(self, prefix: str) -> List[Path]:
        #Найти файлы по префиксу
        return [p for p in self.dir_path.iterdir() if p.is_file() and p.name.startswith(prefix)]

    def analyze_pattern_files(self) -> Dict[str, Dict]:
        # Найти файлы формата 'префикс_№' (с расширением или без)
        # и вывести полную статистику по всем найденным префиксам.
        # Регулярка разделяет имя на "префикс", "номер" и "расширение"
        pattern = re.compile(r"^(.+)_(\d+)(\.[^.]+)?$")
        groups: Dict[str, List[Tuple[int, Path]]] = {}

        for item in self.dir_path.iterdir():
            if item.is_file():
                match = pattern.match(item.name)
                if match:
                    prefix, num_str, _ = match.groups()
                    num = int(num_str)
                    groups.setdefault(prefix, []).append((num, item))

        summary = {}
        for prefix, files in groups.items():
            numbers = sorted([num for num, _ in files])
            summary[prefix] = {
                "count": len(files),
                "min_number": numbers[0] if numbers else None,
                "max_number": numbers[-1] if numbers else None,
                "numbers": numbers,
                "files": [path for _, path in files]
            }

        return summary

    def get_next_available_filename(
        self, 
        prefix: str = "picture_", 
        extension: str = ".png",
        return_full_path: bool = False
    ) -> str | Path:
        """Находит наименьший свободный номер для файла с заданным префиксом."""
        # Нормализуем расширение (добавляем точку, если забыли)
        if extension and not extension.startswith('.'):
            extension = f".{extension}"

        # Регулярное выражение для поиска чисел после префикса
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(extension)}$")
        existing_indices = set()

        # 1. Собираем все имеющиеся индексы
        for file_path in self.dir_path.iterdir():
            if file_path.is_file():
                match = pattern.match(file_path.name)
                if match:
                    existing_indices.add(int(match.group(1)))

        # 2. Ищем наименьший свободный номер, начиная с 1
        next_index = 1
        while next_index in existing_indices:
            next_index += 1

        filename = f"{prefix}{next_index}{extension}"
        
        # Возвращаем либо имя файла (str), либо полный путь (Path)
        return (self.dir_path / filename) if return_full_path else filename


class ComradeManager(DirectoryScanner):
    # Специализированный класс для работы с файлами формата 'префикс_№.ext'.

    def __init__(self, directory_path: str | Path, prefix: str = "comrade"):
        super().__init__(directory_path)
        self.prefix = prefix
        # Ищет файлы вида: <prefix>_<number>.<ext> (расширение опционально)
        self.pattern = re.compile(rf"^{re.escape(self.prefix)}_(\d+)(\.[^.]+)?$")

    def _get_numbered_files(self) -> List[Tuple[int, Path, str]]:
        # Возвращает отсортированный список кортежей: (номер, путь_к_файлу, расширение).
        result = []
        for item in self.dir_path.iterdir():
            if item.is_file():
                match = self.pattern.match(item.name)
                if match:
                    num = int(match.group(1))
                    ext = match.group(2) or ""
                    result.append((num, item, ext))
        return sorted(result, key=lambda x: x[0])

    def find_missing_numbers(self) -> List[int]:
        # Найти пропущенные номера в последовательности
        files = self._get_numbered_files()
        if not files:
            return []

        existing_numbers = {num for num, _, _ in files}
        max_num = files[-1][0]
        min_num = files[0][0]

        # Находим номера, которых не хватает между минимальным и максимальным
        missing = [num for num in range(min_num, max_num + 1) if num not in existing_numbers]
        return missing

    def change_prefix(self, new_prefix: str) -> None:
        # Поменять префикс файлов с сохранением их номеров и расширений
        files = self._get_numbered_files()
        for num, path, ext in files:
            new_name = f"{new_prefix}_{num}{ext}"
            path.rename(self.dir_path / new_name)
        self.prefix = new_prefix
        self.pattern = re.compile(rf"^{re.escape(self.prefix)}_(\d+)(\.[^.]+)?$")

    def compress_and_renumber(self, start_from: int = 1, new_prefix: Optional[str] = None) -> None:
        # "Сжать" последовательность: устраняет дыры в нумерации.
        # Если файлов всего 295, а максимальный номер 300 — переименует их по порядку от start_from до len(files).
        # При необходимости можно одновременно сменить префикс.
        files = self._get_numbered_files()
        target_prefix = new_prefix or self.prefix

        # Чтобы избежать конфликта имён (например, файл 2 переименовывается в 1, когда 1 ещё существует),
        # выполняем переименование через временные имена или сортировку.

        # Шаг 1: переименовываем во временные имена
        temp_files = []
        for idx, (num, path, ext) in enumerate(files):
            temp_name = f"__temp_{idx}_{path.name}"
            temp_path = self.dir_path / temp_name
            path.rename(temp_path)
            temp_files.append((temp_path, ext))

        # Шаг 2: присваиваем итоговые сжатые номера
        for idx, (temp_path, ext) in enumerate(temp_files, start=start_from):
            final_name = f"{target_prefix}_{idx}{ext}"
            temp_path.rename(self.dir_path / final_name)

        if new_prefix:
            self.prefix = new_prefix
            self.pattern = re.compile(rf"^{re.escape(self.prefix)}_(\d+)(\.[^.]+)?$")

    def get_next_filename(self, extension: Optional[str] = None, return_full_path: bool = False) -> str | Path:
        # Использует префикс и расширение, заданные в самом экземпляре класса
        ext = extension or getattr(self, 'default_ext', '.png')
        pref = self.prefix if self.prefix.endswith('_') else f"{self.prefix}_"
        
        return self.get_next_available_filename(
            prefix=pref, 
            extension=ext, 
            return_full_path=return_full_path
        )


