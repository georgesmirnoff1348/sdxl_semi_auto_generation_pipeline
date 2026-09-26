"""Локальный фактический отчёт о датасете (никаких внешних вызовов).

Создаёт два файла (атомарно, через ``*.tmp`` + ``Path.replace``):

* ``dataset_description.md`` — только факты из YAML и манифеста;
* ``dataset_description_request.md`` — opt-in заготовка «перепиши отчёт текстом»
  для ручного копирования в любой чат-агент (сам ``datasetgen`` никого не зовёт).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from . import layout
from .determinism import material
from .manifest import ManifestState
from .planner import background_prompt, settings_for
from .schema import SALT, STATUS_REJECTED, STATUS_SUCCEEDED, JobConfig

LOCAL_PROVENANCE = "Сгенерировано локально командой `datasetgen`; внешние агенты не вызывались."

AGENT_PREAMBLE = (
    "`datasetgen` не выполнял и не будет выполнять внешние команды; "
    "этот файл подготовлен для ручного копирования в любой чат-агент."
)

AGENT_TASK_BLOCK = """## ЗАДАЧА ДЛЯ ВНЕШНЕГО АГЕНТА

Перепиши содержимое отчёта ниже как связный текст для неспециалиста:
объясни, что за датасет, какие в нём классы, как выглядят изображения,
сколько их и для чего такой набор может пригодиться.

Требования к ответу:
- не выдумывай того, чего нет в отчёте (модель, версии, метрики качества);
- сохрани числа (количество изображений по классам, имена классов);
- объясни простыми словами, что означает task и какие есть ограничения;
- ответ на русском языке, объём — примерно один экран текста.
"""


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _counts_by_class(state: ManifestState, job: JobConfig) -> dict[str, dict[str, int]]:
    """Счётчики по эффективному состоянию (последняя запись ключа).

    Считать нужно НЕ все ``succeeded``-попытки: после ``regenerate`` одна и та же
    картинка была произведена дважды, а в каталоге класса лежит один файл, и
    таблица реализаций промптов тоже строится по эффективному состоянию.
    """

    counts = {spec.name: {"requested": spec.count, "produced": 0, "rejected": 0} for spec in job.classes}
    for key, record in state.last.items():
        name = key[0]
        if name not in counts:
            continue
        status = record.get("status")
        if status == STATUS_SUCCEEDED:
            counts[name]["produced"] += 1
        elif status == STATUS_REJECTED:
            counts[name]["rejected"] += 1
    return counts


def _prompt_rows(state: ManifestState, job: JobConfig) -> list[tuple[str, int, int, int, str, str | None]]:
    order = {spec.name: position for position, spec in enumerate(job.classes)}
    rows = []
    for key, record in state.last.items():
        if record.get("status") in (None, "failed"):
            continue
        name, index = key
        attempt = int(record.get("attempt", 1))
        material_text = material(job.seed, job.task, name, index, attempt)
        rows.append(
            (
                name,
                index,
                attempt,
                int(record.get("seed", 0)),
                str(record.get("prompt", "")),
                background_prompt(job, material_text),
            )
        )
    rows.sort(key=lambda row: (order.get(row[0], 99), row[1]))
    return rows


def render_report(
    *,
    job: JobConfig,
    model_id: str,
    root: Path,
    state: ManifestState,
    succeeded: int,
    failed: int,
    rejected_moved: int,
    errors: Sequence[str],
) -> str:
    counts = _counts_by_class(state, job)
    rows = _prompt_rows(state, job)
    limit = job.description.max_prompt_rows
    shown = rows[:limit]

    lines: list[str] = []
    lines.append("# dataset_description")
    lines.append("")
    lines.append(LOCAL_PROVENANCE)
    lines.append("")

    lines.append("## 1. Задача и раскладка")
    lines.append("")
    lines.append(f"- task: `{job.task}`")
    lines.append(f"- generation.mode: `{job.generation.mode}`")
    lines.append(f"- output_dir: `{root}`")
    lines.append("- раскладка файлов:")
    lines.append("  - `<класс>/<класс>_<index>.png` — финальные изображения датасета;")
    lines.append("  - `_rejected/<класс>_<index>_<attempt>.png` — отклонённые изображения (перемещены, не удалены);")
    lines.append("  - `_intermediate/<класс>/<класс>_<index>_<attempt>_subject.png` и `..._cut.png` — промежуточные файлы (только для `two_layer`);")
    lines.append("  - `manifest.jsonl` — append-only журнал попыток (по записи на попытку);")
    lines.append("  - `dataset_description.md` — этот отчёт;")
    lines.append("  - `dataset_description_request.md` — заготовка для внешнего агента (только при `dataset_description.request_file`).")
    lines.append("")

    lines.append("## 2. Классы")
    lines.append("")
    lines.append("| класс | запрошено | произведено | отклонено |")
    lines.append("|---|---|---|---|")
    for spec in job.classes:
        row = counts[spec.name]
        lines.append(f"| `{spec.name}` | {row['requested']} | {row['produced']} | {row['rejected']} |")
    lines.append("")

    lines.append("## 3. Источник промптов")
    lines.append("")
    for name, spec in job.all_prompt_specs():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- template: `{spec.template}`")
        lines.append(f"- negative_prompt: `{spec.negative_prompt}`")
        lines.append("- variables:")
        for variable in spec.variables:
            values = ", ".join(f"`{value}`" for value in spec.variables[variable])
            lines.append(f"  - `{variable}`: {values}")
        lines.append("")

    lines.append("## 4. Реализации промптов")
    lines.append("")
    header = "| класс | index | attempt | seed | prompt |"
    divider = "|---|---|---|---|---|"
    if job.generation.mode == "two_layer":
        header = "| класс | index | attempt | seed | prompt | background prompt |"
        divider = "|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(divider)
    for name, index, attempt, seed, prompt, background in shown:
        label = f"`{name}` | {index}"
        if job.generation.mode == "two_layer":
            lines.append(
                f"| {label} | {attempt} | {seed} | {prompt} | {background or ''} |"
            )
        else:
            lines.append(f"| {label} | {attempt} | {seed} | {prompt} |")
    if len(rows) > len(shown):
        lines.append("")
        lines.append(
            f"показаны первые {len(shown)} из {len(rows)}; полный список — в `manifest.jsonl`"
        )
    lines.append("")

    lines.append("## 5. Параметры генерации")
    lines.append("")
    lines.append(f"- `--model-id`: `{model_id}` (в пайплайн передаётся как `OrdinaryGen(model=...)`)")
    inference = settings_for(job.inference)
    lines.append(
        "- inference: "
        + ", ".join(f"{key}={value}" for key, value in inference.items())
    )
    background = job.generation.background
    if background is not None:
        background_inference = settings_for(background.inference)
        lines.append(f"- generation.background.model_id: `{background.model_id}`")
        lines.append(f"- generation.cutter: `{job.generation.cutter}`")
        lines.append(
            "- generation.background.inference: "
            + ", ".join(f"{key}={value}" for key, value in background_inference.items())
        )
    lines.append("")

    lines.append("## 6. Результаты")
    lines.append("")
    lines.append(f"- successes: {succeeded}")
    lines.append(f"- failures: {failed}")
    lines.append(f"- отклонено (перемещено в `_rejected/` за этот прогон): {rejected_moved}")
    for message in list(errors)[:5]:
        lines.append(f"- error: {message}")
    lines.append("")

    lines.append("## 7. Воспроизводимость")
    lines.append("")
    lines.append(f"- корневой сид задачи: `{job.seed}`")
    lines.append(f"- идентификатор формулы: `{SALT}`")
    lines.append("- повтор той же попытки даёт тот же `prompt` и тот же `seed`;")
    lines.append("- байт-в-байт воспроизводимость пикселей НЕ гарантируется:")
    lines.append("  - revision репозитория модели не закреплена (используется плавающий HEAD);")
    lines.append("  - fp16, attention implementation, версии CUDA/драйвера, CUDA vs MPS, версии torch/diffusers;")
    lines.append("  - sampling-обёртки и планировщики внешних библиотек;")
    lines.append("- отклонённая попытка намеренно никогда не воспроизводится: `attempt` входит в material, поэтому новая попытка получает другой `seed` (и, как правило, другую реализацию промпта);")
    lines.append("- guard сверяет новую пару `(seed, prompt)` со ВСЕМИ прошлыми реализациями индекса, а не только с последней; при совпадении попытка помечается `failed` со `error.stage = \"guard\"` и изображение не перезаписывается;")
    lines.append("- известное ограничение окружения: без CUDA/MPS legacy-обёртки (`OrdinaryGen.__init__`, `ControlNetGen.__init__`) не проставляют `self.device` и падают с `AttributeError`; реальная генерация рассчитана на GPU (Kaggle/Colab);")
    lines.append("")

    lines.append("## 8. Происхождение текста")
    lines.append("")
    lines.append(LOCAL_PROVENANCE)
    if job.description.request_file:
        lines.append("")
        lines.append(
            f"Дополнительно подготовлена заготовка `{layout.description_request_path(root).name}` "
            f"из поля `dataset_description.request_file` (`{job.description.request_file}`): "
            "это ручной шаблон запроса, а не вызов внешнего сервиса."
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(
    *,
    job: JobConfig,
    model_id: str,
    root: Path,
    state: ManifestState,
    succeeded: int,
    failed: int,
    rejected_moved: int,
    errors: Sequence[str],
    warn: Callable[[str], None] | None = None,
) -> bool:
    """Создать ``dataset_description.md`` (+ request-файл). Возвращает True, если записан отчёт."""

    if not job.description.enabled:
        return False
    report = render_report(
        job=job,
        model_id=model_id,
        root=root,
        state=state,
        succeeded=succeeded,
        failed=failed,
        rejected_moved=rejected_moved,
        errors=errors,
    )
    _atomic_write(layout.description_path(root), report)

    request_file = job.description.request_file
    if not request_file:
        return True

    source = Path(request_file)
    if not source.is_file():
        if warn is not None:
            warn(
                f"dataset_description.request_file: '{request_file}' is not a readable file; "
                "dataset_description_request.md was not created"
            )
        return True

    try:
        user_material = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        if warn is not None:
            warn(
                f"dataset_description.request_file: '{request_file}' could not be read ({exc}); "
                "dataset_description_request.md was not created"
            )
        return True

    request = "\n".join(
        [
            "# dataset_description_request",
            "",
            AGENT_PREAMBLE,
            "",
            AGENT_TASK_BLOCK,
            "",
            "## Исходный материал пользователя",
            "",
            f"(файл `{request_file}`, прочитан локально и не изменялся)",
            "",
            "```text",
            user_material.rstrip("\n"),
            "```",
            "",
            "---",
            "",
            "## Фактический отчёт datasetgen",
            "",
            report,
        ]
    )
    _atomic_write(layout.description_request_path(root), request)
    return True
