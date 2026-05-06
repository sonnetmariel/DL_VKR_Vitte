# report_service.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config import AppConfig, get_artifact_status, get_config, get_public_config_summary
from image_utils import SavedImageInfo, read_text_file, save_text_report, write_json_file


class ReportServiceError(RuntimeError):
    """
    Ошибка формирования пользовательского отчета.
    """


@dataclass(frozen=True)
class ProcessingReport:
    """
    Сформированный отчет об обработке изображения.
    """

    title: str
    operation_type: str
    markdown_text: str
    payload_for_llm: dict[str, Any]
    saved_report: SavedImageInfo | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "operation_type": self.operation_type,
            "markdown_text": self.markdown_text,
            "payload_for_llm": self.payload_for_llm,
            "saved_report": asdict(self.saved_report) if self.saved_report else None,
            "created_at": self.created_at,
        }


OPERATION_NAMES_RU = {
    "segmentation": "сегментация и обработка фона",
    "restoration": "восстановление качества изображения",
    "full_pipeline": "полный конвейер восстановления и обработки фона",
}


def object_to_plain_dict(value: Any) -> Any:
    """
    Преобразует объект результата в обычный словарь.

    Поддерживаются словари, списки и dataclass-объекты.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): object_to_plain_dict(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [object_to_plain_dict(item) for item in value]

    if is_dataclass(value):
        return object_to_plain_dict(asdict(value))

    if isinstance(value, Path):
        return str(value)

    return value


def get_nested(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    """
    Безопасно извлекает вложенное значение из словаря.
    """
    current: Any = data

    for key in path:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def format_float(value: Any, digits: int = 4, default: str = "нет данных") -> str:
    """
    Форматирует число для пользовательского отчета.
    """
    if value is None:
        return default

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return default


def format_percent(value: Any, digits: int = 2, default: str = "нет данных") -> str:
    """
    Форматирует процент для пользовательского отчета.
    """
    if value is None:
        return default

    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return default


def format_optional(value: Any, default: str = "нет данных") -> str:
    """
    Форматирует произвольное значение.
    """
    if value is None or value == "":
        return default

    return str(value)


def markdown_escape_table_text(value: Any) -> str:
    """
    Упрощает текст для вставки в таблицу Markdown.
    """
    text = format_optional(value)
    return text.replace("|", "\\|").replace("\n", " ")


def warnings_to_markdown(warnings: list[str] | None) -> str:
    """
    Формирует блок предупреждений.
    """
    if not warnings:
        return "Существенные предупреждения отсутствуют."

    lines = []

    for warning in warnings:
        lines.append(f"- {warning}")

    return "\n".join(lines)


def compact_records_for_llm(records: list[dict[str, Any]], max_records: int = 8) -> list[dict[str, Any]]:
    """
    Ограничивает количество строк для передачи в языковую модель.

    Это снижает расход ключей и не отправляет лишние данные.
    """
    if not records:
        return []

    return records[:max_records]


def build_segmentation_summary_text(segmentation_data: dict[str, Any]) -> str:
    """
    Формирует краткое текстовое описание результата сегментации.
    """
    mode_name = segmentation_data.get("mode_name_ru", "режим сегментации")
    foreground_percent = get_nested(
        segmentation_data,
        ["mask_stats", "foreground_percent"],
        None,
    )

    bbox_width = get_nested(segmentation_data, ["mask_stats", "bbox_width"], None)
    bbox_height = get_nested(segmentation_data, ["mask_stats", "bbox_height"], None)

    dice = get_nested(segmentation_data, ["model_info", "control_dice"], None)
    iou = get_nested(segmentation_data, ["model_info", "control_iou"], None)
    pixel_accuracy = get_nested(
        segmentation_data,
        ["model_info", "control_pixel_accuracy"],
        None,
    )

    return (
        f"Выполнена операция: {mode_name}. "
        f"Модель выделила область объекта площадью {format_percent(foreground_percent)} "
        f"от площади изображения. "
        f"Размер ограничивающей области: {format_optional(bbox_width)} на "
        f"{format_optional(bbox_height)} пикселей. "
        f"Контрольные показатели модели на экспериментальной проверке: "
        f"мера Dice — {format_float(dice)}, пересечение над объединением — "
        f"{format_float(iou)}, попиксельная точность — {format_float(pixel_accuracy)}."
    )


def build_restoration_summary_text(restoration_data: dict[str, Any]) -> str:
    """
    Формирует краткое текстовое описание результата восстановления.
    """
    mode_name = restoration_data.get("mode_name_ru", "режим восстановления")
    degradation_type_name = restoration_data.get(
        "degradation_type_name_ru",
        "тип ухудшения не указан",
    )

    correction_strength = restoration_data.get("applied_correction_strength")

    sharpness_diff = get_nested(
        restoration_data,
        ["quality_summary", "difference", "sharpness"],
        None,
    )

    contrast_diff = get_nested(
        restoration_data,
        ["quality_summary", "difference", "contrast"],
        None,
    )

    best_mode = get_nested(
        restoration_data,
        ["experiment_reference", "best_mode"],
        {},
    )

    reference_ssim = best_mode.get("restored_structure_similarity") if isinstance(best_mode, dict) else None
    reference_psnr = best_mode.get("restored_signal_noise_ratio") if isinstance(best_mode, dict) else None

    return (
        f"Выполнено восстановление качества изображения. Применен режим: {mode_name}. "
        f"Предполагаемый тип ухудшения: {degradation_type_name}. "
        f"Коэффициент нейросетевой правки: {format_float(correction_strength)}. "
        f"По простой оценке без эталона изменение резкости составило "
        f"{format_float(sharpness_diff)}, изменение контраста — {format_float(contrast_diff)}. "
        f"Справочные результаты контрольной проверки второго эксперимента для лучшего режима: "
        f"структурное сходство — {format_float(reference_ssim)}, отношение сигнала к искажению — "
        f"{format_float(reference_psnr)}."
    )


class ReportService:
    """
    Сервис формирования отчетов.

    Отчет строится локально на основе результатов обработки и сохраненных
    экспериментальных показателей. Языковая модель может использовать этот же
    структурированный набор данных, но ее вызов выполняется в отдельном модуле.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    def build_segmentation_report(
        self,
        segmentation_result: Any,
        save: bool = True,
        source_filename: str | None = None,
    ) -> ProcessingReport:
        """
        Формирует отчет по результату сегментации.
        """
        segmentation_data = object_to_plain_dict(segmentation_result)

        if not isinstance(segmentation_data, dict):
            raise ReportServiceError("Результат сегментации имеет неподдерживаемый формат.")

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_filename = source_filename or Path(
            segmentation_data.get("input_path", "segmentation_result.png")
        ).name

        title = "Отчет о сегментации и обработке фона"
        payload = self.build_llm_payload(
            operation_type="segmentation",
            segmentation_result=segmentation_data,
            restoration_result=None,
        )

        markdown_text = self._render_segmentation_markdown(
            segmentation_data=segmentation_data,
            created_at=created_at,
        )

        saved_report = None

        if save:
            saved_report = save_text_report(
                text=markdown_text,
                folder=self.config.runtime.report_folder,
                source_filename=source_filename,
                suffix="segmentation_report",
                config=self.config,
            )

        return ProcessingReport(
            title=title,
            operation_type="segmentation",
            markdown_text=markdown_text,
            payload_for_llm=payload,
            saved_report=saved_report,
            created_at=created_at,
        )

    def build_restoration_report(
        self,
        restoration_result: Any,
        save: bool = True,
        source_filename: str | None = None,
    ) -> ProcessingReport:
        """
        Формирует отчет по результату восстановления качества.
        """
        restoration_data = object_to_plain_dict(restoration_result)

        if not isinstance(restoration_data, dict):
            raise ReportServiceError("Результат восстановления имеет неподдерживаемый формат.")

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_filename = source_filename or Path(
            restoration_data.get("input_path", "restoration_result.png")
        ).name

        title = "Отчет о восстановлении качества изображения"
        payload = self.build_llm_payload(
            operation_type="restoration",
            segmentation_result=None,
            restoration_result=restoration_data,
        )

        markdown_text = self._render_restoration_markdown(
            restoration_data=restoration_data,
            created_at=created_at,
        )

        saved_report = None

        if save:
            saved_report = save_text_report(
                text=markdown_text,
                folder=self.config.runtime.report_folder,
                source_filename=source_filename,
                suffix="restoration_report",
                config=self.config,
            )

        return ProcessingReport(
            title=title,
            operation_type="restoration",
            markdown_text=markdown_text,
            payload_for_llm=payload,
            saved_report=saved_report,
            created_at=created_at,
        )

    def build_full_pipeline_report(
        self,
        restoration_result: Any,
        segmentation_result: Any,
        save: bool = True,
        source_filename: str | None = None,
    ) -> ProcessingReport:
        """
        Формирует отчет по полному конвейеру.

        Полный конвейер объединяет восстановление качества и дальнейшую обработку фона.
        """
        restoration_data = object_to_plain_dict(restoration_result)
        segmentation_data = object_to_plain_dict(segmentation_result)

        if not isinstance(restoration_data, dict):
            raise ReportServiceError("Результат восстановления имеет неподдерживаемый формат.")

        if not isinstance(segmentation_data, dict):
            raise ReportServiceError("Результат сегментации имеет неподдерживаемый формат.")

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_filename = source_filename or Path(
            restoration_data.get("input_path", segmentation_data.get("input_path", "pipeline_result.png"))
        ).name

        title = "Отчет о полном конвейере нейросетевой обработки"
        payload = self.build_llm_payload(
            operation_type="full_pipeline",
            segmentation_result=segmentation_data,
            restoration_result=restoration_data,
        )

        markdown_text = self._render_full_pipeline_markdown(
            restoration_data=restoration_data,
            segmentation_data=segmentation_data,
            created_at=created_at,
        )

        saved_report = None

        if save:
            saved_report = save_text_report(
                text=markdown_text,
                folder=self.config.runtime.report_folder,
                source_filename=source_filename,
                suffix="full_pipeline_report",
                config=self.config,
            )

        return ProcessingReport(
            title=title,
            operation_type="full_pipeline",
            markdown_text=markdown_text,
            payload_for_llm=payload,
            saved_report=saved_report,
            created_at=created_at,
        )

    def build_llm_payload(
        self,
        operation_type: str,
        segmentation_result: dict[str, Any] | None = None,
        restoration_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Формирует компактный набор данных для языковой модели.

        Изображения не включаются. Передаются только численные показатели,
        режимы и предупреждения. Такой подход снижает расход ключей.
        """
        operation_name = OPERATION_NAMES_RU.get(operation_type, operation_type)

        payload: dict[str, Any] = {
            "operation_type": operation_type,
            "operation_name_ru": operation_name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "instruction": (
                "Сформируй краткий пользовательский отчет на русском языке. "
                "Не выдумывай чисел. Опирайся только на переданные показатели. "
                "Отдельно укажи ограничения результата."
            ),
        }

        if segmentation_result:
            payload["segmentation"] = self._compact_segmentation_payload(segmentation_result)

        if restoration_result:
            payload["restoration"] = self._compact_restoration_payload(restoration_result)

        return payload

    def save_llm_payload(
        self,
        payload: dict[str, Any],
        source_filename: str,
        suffix: str = "llm_payload",
    ) -> Path:
        """
        Сохраняет структурированные данные для языковой модели в JSON.
        """
        filename = Path(source_filename).stem + f"_{suffix}.json"
        path = self.config.runtime.report_folder / filename

        return write_json_file(payload, path)

    def build_about_report_data(self) -> dict[str, Any]:
        """
        Возвращает сведения для страницы о системе.
        """
        example_restoration_report = read_text_file(
            self.config.restoration.example_report_path,
            default="",
        )

        return {
            "app": get_public_config_summary(self.config),
            "artifact_status": get_artifact_status(self.config),
            "restoration_example_report": example_restoration_report,
        }

    def _compact_segmentation_payload(
        self,
        segmentation_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Сжимает результат сегментации для внешнего текстового объяснения.
        """
        return {
            "mode_name_ru": segmentation_data.get("mode_name_ru"),
            "mask_stats": segmentation_data.get("mask_stats", {}),
            "warnings": segmentation_data.get("warnings", []),
            "model_info": {
                "model_name": get_nested(segmentation_data, ["model_info", "model_name"]),
                "architecture": get_nested(segmentation_data, ["model_info", "architecture"]),
                "encoder": get_nested(segmentation_data, ["model_info", "encoder"]),
                "control_dice": get_nested(segmentation_data, ["model_info", "control_dice"]),
                "control_iou": get_nested(segmentation_data, ["model_info", "control_iou"]),
                "control_pixel_accuracy": get_nested(
                    segmentation_data,
                    ["model_info", "control_pixel_accuracy"],
                ),
            },
            "summary": build_segmentation_summary_text(segmentation_data),
        }

    def _compact_restoration_payload(
        self,
        restoration_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Сжимает результат восстановления для внешнего текстового объяснения.
        """
        experiment_reference = restoration_data.get("experiment_reference", {})

        final_summary = []
        adaptive_by_type_summary = []

        if isinstance(experiment_reference, dict):
            final_summary = compact_records_for_llm(
                experiment_reference.get("final_summary", []),
                max_records=4,
            )
            adaptive_by_type_summary = compact_records_for_llm(
                experiment_reference.get("adaptive_by_type_summary", []),
                max_records=7,
            )

        return {
            "mode_name_ru": restoration_data.get("mode_name_ru"),
            "degradation_type_name_ru": restoration_data.get("degradation_type_name_ru"),
            "applied_correction_strength": restoration_data.get("applied_correction_strength"),
            "quality_summary_without_reference": restoration_data.get("quality_summary", {}),
            "warnings": restoration_data.get("warnings", []),
            "experiment_reference": {
                "final_summary": final_summary,
                "adaptive_by_type_summary": adaptive_by_type_summary,
                "important_note": experiment_reference.get("important_note")
                if isinstance(experiment_reference, dict)
                else None,
            },
            "summary": build_restoration_summary_text(restoration_data),
        }

    def _render_segmentation_markdown(
        self,
        segmentation_data: dict[str, Any],
        created_at: str,
    ) -> str:
        """
        Создает Markdown-отчет по сегментации.
        """
        mode_name = segmentation_data.get("mode_name_ru", "сегментация")
        result_image = get_nested(segmentation_data, ["result_image", "relative_path"], "")
        mask_image = get_nested(segmentation_data, ["mask_image", "relative_path"], "")
        preview_image = get_nested(segmentation_data, ["preview_image", "relative_path"], "")

        foreground_percent = get_nested(
            segmentation_data,
            ["mask_stats", "foreground_percent"],
            None,
        )

        bbox_left = get_nested(segmentation_data, ["mask_stats", "bbox_left"], None)
        bbox_top = get_nested(segmentation_data, ["mask_stats", "bbox_top"], None)
        bbox_width = get_nested(segmentation_data, ["mask_stats", "bbox_width"], None)
        bbox_height = get_nested(segmentation_data, ["mask_stats", "bbox_height"], None)

        model_name = get_nested(segmentation_data, ["model_info", "model_name"], "модель сегментации")
        dice = get_nested(segmentation_data, ["model_info", "control_dice"], None)
        iou = get_nested(segmentation_data, ["model_info", "control_iou"], None)
        pixel_accuracy = get_nested(segmentation_data, ["model_info", "control_pixel_accuracy"], None)

        warnings = segmentation_data.get("warnings", [])

        return f"""# Отчет о сегментации и обработке фона

Дата формирования: {created_at}

## Выполненная операция

{build_segmentation_summary_text(segmentation_data)}

## Основные сведения

| Показатель | Значение |
|---|---:|
| Режим обработки | {markdown_escape_table_text(mode_name)} |
| Доля выделенного объекта | {format_percent(foreground_percent)} |
| Левая граница области | {format_optional(bbox_left)} |
| Верхняя граница области | {format_optional(bbox_top)} |
| Ширина области | {format_optional(bbox_width)} |
| Высота области | {format_optional(bbox_height)} |

## Сведения о модели

| Показатель | Значение |
|---|---:|
| Название модели | {markdown_escape_table_text(model_name)} |
| Мера Dice на контрольной проверке | {format_float(dice)} |
| Пересечение над объединением | {format_float(iou)} |
| Попиксельная точность | {format_float(pixel_accuracy)} |

## Файлы результата

| Файл | Адрес |
|---|---|
| Результат обработки | {markdown_escape_table_text(result_image)} |
| Маска объекта | {markdown_escape_table_text(mask_image)} |
| Сравнительное изображение | {markdown_escape_table_text(preview_image)} |

## Предупреждения и ограничения

{warnings_to_markdown(warnings)}

## Интерпретация

Результат сегментации следует оценивать визуально, поскольку границы объекта могут быть неточными на сложном фоне, при перекрытии объекта или при сильном отличии изображения от обучающих примеров. При корректной маске результат можно использовать для удаления, размытия или замены фона.
"""

    def _render_restoration_markdown(
        self,
        restoration_data: dict[str, Any],
        created_at: str,
    ) -> str:
        """
        Создает Markdown-отчет по восстановлению качества.
        """
        mode_name = restoration_data.get("mode_name_ru", "режим восстановления")
        degradation_type_name = restoration_data.get(
            "degradation_type_name_ru",
            "тип ухудшения не указан",
        )
        correction_strength = restoration_data.get("applied_correction_strength")

        result_image = get_nested(restoration_data, ["result_image", "relative_path"], "")
        preview_image = get_nested(restoration_data, ["preview_image", "relative_path"], "")

        before = get_nested(restoration_data, ["quality_summary", "before"], {})
        after = get_nested(restoration_data, ["quality_summary", "after"], {})
        difference = get_nested(restoration_data, ["quality_summary", "difference"], {})

        best_mode = get_nested(restoration_data, ["experiment_reference", "best_mode"], {})
        reference_note = get_nested(
            restoration_data,
            ["experiment_reference", "important_note"],
            "",
        )

        warnings = restoration_data.get("warnings", [])

        return f"""# Отчет о восстановлении качества изображения

Дата формирования: {created_at}

## Выполненная операция

{build_restoration_summary_text(restoration_data)}

## Режим обработки

| Показатель | Значение |
|---|---:|
| Режим восстановления | {markdown_escape_table_text(mode_name)} |
| Предполагаемый тип ухудшения | {markdown_escape_table_text(degradation_type_name)} |
| Коэффициент нейросетевой правки | {format_float(correction_strength)} |

## Простые признаки изображения без эталона

| Признак | До обработки | После обработки | Изменение |
|---|---:|---:|---:|
| Яркость | {format_float(before.get("brightness"))} | {format_float(after.get("brightness"))} | {format_float(difference.get("brightness"))} |
| Контраст | {format_float(before.get("contrast"))} | {format_float(after.get("contrast"))} | {format_float(difference.get("contrast"))} |
| Резкость | {format_float(before.get("sharpness"))} | {format_float(after.get("sharpness"))} | {format_float(difference.get("sharpness"))} |
| Цветовая насыщенность | {format_float(before.get("colorfulness"))} | {format_float(after.get("colorfulness"))} | {format_float(difference.get("colorfulness"))} |

## Справка по результатам второго эксперимента

| Показатель контрольной проверки | Значение |
|---|---:|
| Лучший режим | {markdown_escape_table_text(best_mode.get("mode_name", "нет данных") if isinstance(best_mode, dict) else "нет данных")} |
| Отношение сигнала к искажению | {format_float(best_mode.get("restored_signal_noise_ratio") if isinstance(best_mode, dict) else None)} |
| Структурное сходство | {format_float(best_mode.get("restored_structure_similarity") if isinstance(best_mode, dict) else None)} |
| Средняя абсолютная ошибка | {format_float(best_mode.get("restored_mean_absolute_error") if isinstance(best_mode, dict) else None)} |

{reference_note}

## Файлы результата

| Файл | Адрес |
|---|---|
| Восстановленное изображение | {markdown_escape_table_text(result_image)} |
| Сравнительное изображение | {markdown_escape_table_text(preview_image)} |

## Предупреждения и ограничения

{warnings_to_markdown(warnings)}

## Интерпретация

В пользовательском приложении нет эталонного изображения, поэтому показатели структурного сходства, отношения сигнала к искажению и средней абсолютной ошибки для загруженного пользователем файла не рассчитываются. Эти показатели приводятся только как справочные результаты контрольной проверки второго эксперимента. Для пользовательского файла рассчитываются простые признаки изображения без эталона.
"""

    def _render_full_pipeline_markdown(
        self,
        restoration_data: dict[str, Any],
        segmentation_data: dict[str, Any],
        created_at: str,
    ) -> str:
        """
        Создает Markdown-отчет по полному конвейеру.
        """
        restoration_part = build_restoration_summary_text(restoration_data)
        segmentation_part = build_segmentation_summary_text(segmentation_data)

        restoration_warnings = restoration_data.get("warnings", [])
        segmentation_warnings = segmentation_data.get("warnings", [])
        all_warnings = restoration_warnings + segmentation_warnings

        restoration_result_image = get_nested(
            restoration_data,
            ["result_image", "relative_path"],
            "",
        )
        segmentation_result_image = get_nested(
            segmentation_data,
            ["result_image", "relative_path"],
            "",
        )
        segmentation_mask_image = get_nested(
            segmentation_data,
            ["mask_image", "relative_path"],
            "",
        )

        return f"""# Отчет о полном конвейере нейросетевой обработки изображения

Дата формирования: {created_at}

## Назначение обработки

Полный конвейер объединяет два результата магистерского исследования: восстановление качества изображения и сегментацию объекта животного. Сначала изображение проходит нейросетевое восстановление, затем обработанное изображение передается в модель сегментации для выделения объекта и работы с фоном.

## Этап восстановления качества

{restoration_part}

## Этап сегментации и обработки фона

{segmentation_part}

## Файлы результата

| Файл | Адрес |
|---|---|
| Восстановленное изображение | {markdown_escape_table_text(restoration_result_image)} |
| Итоговое изображение после сегментации | {markdown_escape_table_text(segmentation_result_image)} |
| Маска объекта | {markdown_escape_table_text(segmentation_mask_image)} |

## Предупреждения и ограничения

{warnings_to_markdown(all_warnings)}

## Интерпретация

Результат полного конвейера следует рассматривать как последовательную нейросетевую обработку изображения. Модуль восстановления улучшает качество входного изображения в пределах обученной постановки, а модуль сегментации использует полученное изображение для выделения объекта. Качество итогового результата зависит от исходного изображения, характера искажения, положения животного в кадре и сложности фона.
"""


def get_report_service() -> ReportService:
    """
    Создает сервис отчетов.
    """
    return ReportService()


def build_and_save_report(
    operation_type: str,
    source_filename: str,
    restoration_result: Any | None = None,
    segmentation_result: Any | None = None,
) -> ProcessingReport:
    """
    Удобная функция для создания отчета из app.py.
    """
    service = get_report_service()

    if operation_type == "segmentation":
        if segmentation_result is None:
            raise ReportServiceError("Для отчета сегментации не передан результат сегментации.")

        return service.build_segmentation_report(
            segmentation_result=segmentation_result,
            save=True,
            source_filename=source_filename,
        )

    if operation_type == "restoration":
        if restoration_result is None:
            raise ReportServiceError("Для отчета восстановления не передан результат восстановления.")

        return service.build_restoration_report(
            restoration_result=restoration_result,
            save=True,
            source_filename=source_filename,
        )

    if operation_type == "full_pipeline":
        if restoration_result is None or segmentation_result is None:
            raise ReportServiceError(
                "Для отчета полного конвейера нужны результаты восстановления и сегментации."
            )

        return service.build_full_pipeline_report(
            restoration_result=restoration_result,
            segmentation_result=segmentation_result,
            save=True,
            source_filename=source_filename,
        )

    raise ReportServiceError(f"Неизвестный тип операции: {operation_type}")
