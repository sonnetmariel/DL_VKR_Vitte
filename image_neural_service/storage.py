# storage.py
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config import AppConfig, get_config
from image_utils import ensure_directory


class StorageError(RuntimeError):
    """
    Ошибка работы с хранилищем истории обработки.
    """


@dataclass(frozen=True)
class HistoryRecord:
    """
    Запись истории обработки изображения.
    """

    id: int | None
    record_uid: str
    created_at: str
    operation_type: str
    operation_name_ru: str
    status: str
    source_filename: str
    input_path: str
    input_url: str
    result_path: str
    result_url: str
    preview_path: str
    preview_url: str
    report_path: str
    report_url: str
    llm_used: bool
    llm_success: bool
    short_summary: str
    data: dict[str, Any]
    error_text: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StorageStats:
    """
    Сводка по истории обработки.
    """

    total_count: int
    success_count: int
    error_count: int
    segmentation_count: int
    restoration_count: int
    full_pipeline_count: int
    llm_success_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OPERATION_NAMES_RU = {
    "segmentation": "сегментация и обработка фона",
    "restoration": "восстановление качества изображения",
    "full_pipeline": "полный конвейер восстановления и обработки фона",
}


def now_text() -> str:
    """
    Возвращает текущую дату в текстовом виде.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_record_uid() -> str:
    """
    Создает уникальный идентификатор записи истории.
    """
    return "rec_" + uuid.uuid4().hex


def object_to_plain_dict(value: Any) -> Any:
    """
    Преобразует объект результата в обычный словарь.

    Поддерживаются dataclass-объекты, словари, списки, пути и простые значения.
    """
    if value is None:
        return None

    if is_dataclass(value):
        return object_to_plain_dict(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): object_to_plain_dict(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [object_to_plain_dict(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value


def json_dumps_safe(data: Any) -> str:
    """
    Сериализует данные в JSON.
    """
    return json.dumps(
        object_to_plain_dict(data),
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def json_loads_safe(text: str | None, default: Any = None) -> Any:
    """
    Читает JSON-строку.
    """
    if default is None:
        default = {}

    if not text:
        return default

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def get_nested(data: dict[str, Any], path: list[str], default: Any = "") -> Any:
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


def first_not_empty(*values: Any, default: str = "") -> str:
    """
    Возвращает первое непустое значение в виде строки.
    """
    for value in values:
        if value is None:
            continue

        text = str(value)

        if text:
            return text

    return default


def extract_saved_file_data(data: dict[str, Any], key: str) -> tuple[str, str]:
    """
    Извлекает путь и адрес сохраненного файла из вложенного результата.

    Поддерживаются объекты вида:
    result_image.absolute_path
    result_image.relative_path
    """
    file_data = data.get(key, {})

    if not isinstance(file_data, dict):
        return "", ""

    path = first_not_empty(
        file_data.get("absolute_path"),
        file_data.get("path"),
        default="",
    )

    url = first_not_empty(
        file_data.get("relative_path"),
        file_data.get("url"),
        default="",
    )

    return path, url


def extract_primary_result_data(
    operation_type: str,
    result_data: dict[str, Any],
) -> tuple[str, str, str, str]:
    """
    Извлекает главный результат и предварительное изображение.

    Для простых операций структура одинакова. Для полного конвейера результатом
    считается итог сегментации после восстановления.
    """
    result_path, result_url = extract_saved_file_data(result_data, "result_image")
    preview_path, preview_url = extract_saved_file_data(result_data, "preview_image")

    if operation_type == "full_pipeline":
        segmentation = result_data.get("segmentation_result", {})
        restoration = result_data.get("restoration_result", {})

        if isinstance(segmentation, dict):
            seg_result_path, seg_result_url = extract_saved_file_data(
                segmentation,
                "result_image",
            )
            seg_preview_path, seg_preview_url = extract_saved_file_data(
                segmentation,
                "preview_image",
            )

            result_path = seg_result_path or result_path
            result_url = seg_result_url or result_url
            preview_path = seg_preview_path or preview_path
            preview_url = seg_preview_url or preview_url

        if not preview_url and isinstance(restoration, dict):
            rest_preview_path, rest_preview_url = extract_saved_file_data(
                restoration,
                "preview_image",
            )
            preview_path = rest_preview_path or preview_path
            preview_url = rest_preview_url or preview_url

    return result_path, result_url, preview_path, preview_url


def extract_report_data(report_data: dict[str, Any] | None) -> tuple[str, str]:
    """
    Извлекает путь и адрес сохраненного текстового отчета.
    """
    if not report_data:
        return "", ""

    saved_report = report_data.get("saved_report", {})

    if not isinstance(saved_report, dict):
        return "", ""

    report_path = first_not_empty(
        saved_report.get("absolute_path"),
        saved_report.get("path"),
        default="",
    )

    report_url = first_not_empty(
        saved_report.get("relative_path"),
        saved_report.get("url"),
        default="",
    )

    return report_path, report_url


def extract_input_data(
    result_data: dict[str, Any],
    fallback_path: str = "",
    fallback_url: str = "",
) -> tuple[str, str]:
    """
    Извлекает путь и адрес исходного изображения.
    """
    input_path = first_not_empty(
        result_data.get("input_path"),
        fallback_path,
        default="",
    )

    input_url = first_not_empty(
        result_data.get("input_url"),
        fallback_url,
        default="",
    )

    if not input_path and "restoration_result" in result_data:
        restoration = result_data.get("restoration_result", {})
        if isinstance(restoration, dict):
            input_path = first_not_empty(restoration.get("input_path"), input_path)
            input_url = first_not_empty(restoration.get("input_url"), input_url)

    if not input_path and "segmentation_result" in result_data:
        segmentation = result_data.get("segmentation_result", {})
        if isinstance(segmentation, dict):
            input_path = first_not_empty(segmentation.get("input_path"), input_path)
            input_url = first_not_empty(segmentation.get("input_url"), input_url)

    return input_path, input_url


def build_short_summary(
    operation_type: str,
    result_data: dict[str, Any],
    report_data: dict[str, Any] | None = None,
) -> str:
    """
    Формирует короткое описание записи истории.
    """
    operation_name = OPERATION_NAMES_RU.get(operation_type, operation_type)

    if operation_type == "segmentation":
        mode_name = result_data.get("mode_name_ru", "")
        foreground_percent = get_nested(
            result_data,
            ["mask_stats", "foreground_percent"],
            "",
        )

        return (
            f"{operation_name}: {mode_name}. "
            f"Доля выделенного объекта: {foreground_percent}%."
        )

    if operation_type == "restoration":
        mode_name = result_data.get("mode_name_ru", "")
        degradation_name = result_data.get("degradation_type_name_ru", "")
        correction_strength = result_data.get("applied_correction_strength", "")

        return (
            f"{operation_name}: {mode_name}. "
            f"Тип ухудшения: {degradation_name}. "
            f"Коэффициент правки: {correction_strength}."
        )

    if operation_type == "full_pipeline":
        restoration = result_data.get("restoration_result", {})
        segmentation = result_data.get("segmentation_result", {})

        restoration_mode = ""
        segmentation_mode = ""

        if isinstance(restoration, dict):
            restoration_mode = restoration.get("mode_name_ru", "")

        if isinstance(segmentation, dict):
            segmentation_mode = segmentation.get("mode_name_ru", "")

        return (
            f"{operation_name}: восстановление — {restoration_mode}; "
            f"обработка фона — {segmentation_mode}."
        )

    if report_data and report_data.get("title"):
        return str(report_data["title"])

    return operation_name


def row_to_history_record(row: sqlite3.Row) -> HistoryRecord:
    """
    Преобразует строку SQLite в объект HistoryRecord.
    """
    data = json_loads_safe(row["data_json"], default={})

    return HistoryRecord(
        id=row["id"],
        record_uid=row["record_uid"],
        created_at=row["created_at"],
        operation_type=row["operation_type"],
        operation_name_ru=row["operation_name_ru"],
        status=row["status"],
        source_filename=row["source_filename"] or "",
        input_path=row["input_path"] or "",
        input_url=row["input_url"] or "",
        result_path=row["result_path"] or "",
        result_url=row["result_url"] or "",
        preview_path=row["preview_path"] or "",
        preview_url=row["preview_url"] or "",
        report_path=row["report_path"] or "",
        report_url=row["report_url"] or "",
        llm_used=bool(row["llm_used"]),
        llm_success=bool(row["llm_success"]),
        short_summary=row["short_summary"] or "",
        data=data,
        error_text=row["error_text"],
    )


class StorageService:
    """
    Сервис хранения истории обработки изображений.

    Используется локальная база SQLite. Хранилище не содержит самих изображений,
    а сохраняет пути к файлам в каталоге runtime.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.database_path = self.config.runtime.database_path
        ensure_directory(self.database_path.parent)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        """
        Создает подключение к базе данных.
        """
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        return connection

    def init_db(self) -> None:
        """
        Создает таблицы истории, если они еще не существуют.
        """
        ensure_directory(self.database_path.parent)

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_uid TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    operation_name_ru TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_filename TEXT,
                    input_path TEXT,
                    input_url TEXT,
                    result_path TEXT,
                    result_url TEXT,
                    preview_path TEXT,
                    preview_url TEXT,
                    report_path TEXT,
                    report_url TEXT,
                    llm_used INTEGER NOT NULL DEFAULT 0,
                    llm_success INTEGER NOT NULL DEFAULT 0,
                    short_summary TEXT,
                    data_json TEXT NOT NULL,
                    error_text TEXT
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processing_history_created_at
                ON processing_history(created_at)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processing_history_operation_type
                ON processing_history(operation_type)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processing_history_status
                ON processing_history(status)
                """
            )

            connection.commit()

    def save_processing_record(
        self,
        operation_type: str,
        source_filename: str,
        result: Any,
        report: Any | None = None,
        llm_result: Any | None = None,
        status: str = "success",
        error_text: str | None = None,
        extra: dict[str, Any] | None = None,
        input_path: str = "",
        input_url: str = "",
    ) -> HistoryRecord:
        """
        Сохраняет запись об успешной или частично успешной обработке.

        result может быть результатом сегментации, восстановления или словарем
        с ключами restoration_result и segmentation_result для полного конвейера.
        """
        operation_type = str(operation_type)
        operation_name_ru = OPERATION_NAMES_RU.get(operation_type, operation_type)

        result_data = object_to_plain_dict(result)

        if not isinstance(result_data, dict):
            result_data = {"value": result_data}

        report_data = object_to_plain_dict(report)
        if report_data is not None and not isinstance(report_data, dict):
            report_data = {"value": report_data}

        llm_data = object_to_plain_dict(llm_result)
        if llm_data is not None and not isinstance(llm_data, dict):
            llm_data = {"value": llm_data}

        extracted_input_path, extracted_input_url = extract_input_data(
            result_data,
            fallback_path=input_path,
            fallback_url=input_url,
        )

        result_path, result_url, preview_path, preview_url = extract_primary_result_data(
            operation_type=operation_type,
            result_data=result_data,
        )

        report_path, report_url = extract_report_data(report_data)

        llm_used = bool(llm_data)
        llm_success = bool(llm_data.get("success")) if isinstance(llm_data, dict) else False

        data: dict[str, Any] = {
            "result": result_data,
            "report": report_data,
            "llm_result": llm_data,
            "extra": extra or {},
        }

        short_summary = build_short_summary(
            operation_type=operation_type,
            result_data=result_data,
            report_data=report_data,
        )

        record_uid = make_record_uid()
        created_at = now_text()

        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO processing_history (
                    record_uid,
                    created_at,
                    operation_type,
                    operation_name_ru,
                    status,
                    source_filename,
                    input_path,
                    input_url,
                    result_path,
                    result_url,
                    preview_path,
                    preview_url,
                    report_path,
                    report_url,
                    llm_used,
                    llm_success,
                    short_summary,
                    data_json,
                    error_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_uid,
                    created_at,
                    operation_type,
                    operation_name_ru,
                    status,
                    source_filename,
                    extracted_input_path,
                    extracted_input_url,
                    result_path,
                    result_url,
                    preview_path,
                    preview_url,
                    report_path,
                    report_url,
                    int(llm_used),
                    int(llm_success),
                    short_summary,
                    json_dumps_safe(data),
                    error_text,
                ),
            )

            connection.commit()
            record_id = int(cursor.lastrowid)

        saved = self.get_record(record_id)

        if saved is None:
            raise StorageError("Запись была сохранена, но не найдена при повторном чтении.")

        return saved

    def save_error_record(
        self,
        operation_type: str,
        source_filename: str,
        error_text: str,
        input_path: str = "",
        input_url: str = "",
        extra: dict[str, Any] | None = None,
    ) -> HistoryRecord:
        """
        Сохраняет запись об ошибке обработки.
        """
        result_data = {
            "input_path": input_path,
            "input_url": input_url,
            "error": error_text,
        }

        return self.save_processing_record(
            operation_type=operation_type,
            source_filename=source_filename,
            result=result_data,
            report=None,
            llm_result=None,
            status="error",
            error_text=error_text,
            extra=extra,
            input_path=input_path,
            input_url=input_url,
        )

    def get_record(self, record_id: int) -> HistoryRecord | None:
        """
        Возвращает запись истории по числовому идентификатору.
        """
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM processing_history
                WHERE id = ?
                """,
                (record_id,),
            ).fetchone()

        if row is None:
            return None

        return row_to_history_record(row)

    def get_record_by_uid(self, record_uid: str) -> HistoryRecord | None:
        """
        Возвращает запись истории по уникальному текстовому идентификатору.
        """
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM processing_history
                WHERE record_uid = ?
                """,
                (record_uid,),
            ).fetchone()

        if row is None:
            return None

        return row_to_history_record(row)

    def list_records(
        self,
        limit: int = 30,
        offset: int = 0,
        operation_type: str | None = None,
        status: str | None = None,
    ) -> list[HistoryRecord]:
        """
        Возвращает список записей истории.

        Новые записи идут первыми.
        """
        conditions: list[str] = []
        parameters: list[Any] = []

        if operation_type:
            conditions.append("operation_type = ?")
            parameters.append(operation_type)

        if status:
            conditions.append("status = ?")
            parameters.append(status)

        where_sql = ""

        if conditions:
            where_sql = "WHERE " + " AND ".join(conditions)

        parameters.extend([int(limit), int(offset)])

        query = f"""
            SELECT *
            FROM processing_history
            {where_sql}
            ORDER BY id DESC
            LIMIT ?
            OFFSET ?
        """

        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        return [row_to_history_record(row) for row in rows]

    def count_records(
        self,
        operation_type: str | None = None,
        status: str | None = None,
    ) -> int:
        """
        Считает количество записей истории.
        """
        conditions: list[str] = []
        parameters: list[Any] = []

        if operation_type:
            conditions.append("operation_type = ?")
            parameters.append(operation_type)

        if status:
            conditions.append("status = ?")
            parameters.append(status)

        where_sql = ""

        if conditions:
            where_sql = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT COUNT(*) AS count
            FROM processing_history
            {where_sql}
        """

        with self.connect() as connection:
            row = connection.execute(query, parameters).fetchone()

        return int(row["count"])

    def delete_record(self, record_id: int) -> bool:
        """
        Удаляет запись истории из базы.

        Сами файлы изображений не удаляются, чтобы случайно не потерять результат.
        """
        with self.connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM processing_history
                WHERE id = ?
                """,
                (record_id,),
            )
            connection.commit()

        return cursor.rowcount > 0

    def clear_history(self) -> int:
        """
        Очищает историю обработки.

        Возвращает количество удаленных записей. Файлы runtime не удаляются.
        """
        total_count = self.count_records()

        with self.connect() as connection:
            connection.execute("DELETE FROM processing_history")
            connection.commit()

        return total_count

    def get_stats(self) -> StorageStats:
        """
        Возвращает сводку по истории обработки.
        """
        with self.connect() as connection:
            total_count = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM processing_history"
                ).fetchone()["count"]
            )

            success_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM processing_history
                    WHERE status = 'success'
                    """
                ).fetchone()["count"]
            )

            error_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM processing_history
                    WHERE status = 'error'
                    """
                ).fetchone()["count"]
            )

            segmentation_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM processing_history
                    WHERE operation_type = 'segmentation'
                    """
                ).fetchone()["count"]
            )

            restoration_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM processing_history
                    WHERE operation_type = 'restoration'
                    """
                ).fetchone()["count"]
            )

            full_pipeline_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM processing_history
                    WHERE operation_type = 'full_pipeline'
                    """
                ).fetchone()["count"]
            )

            llm_success_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM processing_history
                    WHERE llm_success = 1
                    """
                ).fetchone()["count"]
            )

        return StorageStats(
            total_count=total_count,
            success_count=success_count,
            error_count=error_count,
            segmentation_count=segmentation_count,
            restoration_count=restoration_count,
            full_pipeline_count=full_pipeline_count,
            llm_success_count=llm_success_count,
        )

    def export_history_json(self, output_path: str | Path | None = None) -> Path:
        """
        Выгружает историю обработки в JSON-файл.
        """
        if output_path is None:
            output_path = self.config.runtime.report_folder / "processing_history_export.json"

        output_path = Path(output_path)
        ensure_directory(output_path.parent)

        records = [
            record.to_dict()
            for record in self.list_records(limit=10_000, offset=0)
        ]

        export_data = {
            "created_at": now_text(),
            "record_count": len(records),
            "records": records,
        }

        output_path.write_text(
            json_dumps_safe(export_data),
            encoding="utf-8",
        )

        return output_path

    def vacuum(self) -> None:
        """
        Оптимизирует файл базы данных.
        """
        with self.connect() as connection:
            connection.execute("VACUUM")
            connection.commit()


def get_storage_service() -> StorageService:
    """
    Создает сервис истории обработки.
    """
    return StorageService()


def init_db() -> None:
    """
    Инициализирует базу истории.
    """
    get_storage_service().init_db()


def save_processing_record(
    operation_type: str,
    source_filename: str,
    result: Any,
    report: Any | None = None,
    llm_result: Any | None = None,
    status: str = "success",
    error_text: str | None = None,
    extra: dict[str, Any] | None = None,
    input_path: str = "",
    input_url: str = "",
) -> HistoryRecord:
    """
    Удобная функция сохранения записи истории.
    """
    return get_storage_service().save_processing_record(
        operation_type=operation_type,
        source_filename=source_filename,
        result=result,
        report=report,
        llm_result=llm_result,
        status=status,
        error_text=error_text,
        extra=extra,
        input_path=input_path,
        input_url=input_url,
    )


def list_history(
    limit: int = 30,
    offset: int = 0,
    operation_type: str | None = None,
    status: str | None = None,
) -> list[HistoryRecord]:
    """
    Удобная функция получения истории.
    """
    return get_storage_service().list_records(
        limit=limit,
        offset=offset,
        operation_type=operation_type,
        status=status,
    )


def get_history_stats() -> StorageStats:
    """
    Удобная функция получения статистики истории.
    """
    return get_storage_service().get_stats()
