# config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"


def _load_env_file(env_path: Path) -> None:
    """
    Загружает значения из файла .env.

    Если установлена библиотека python-dotenv, используется она.
    Если библиотека отсутствует, применяется простой встроенный разбор файла.
    """
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except Exception:
        pass

    with env_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file(ENV_PATH)


def _clean_env_value(value: str | None) -> str:
    if value is None:
        return ""

    return value.strip().strip('"').strip("'")


def _get_first_str(names: str | tuple[str, ...] | list[str], default: str = "") -> str:
    """
    Возвращает первое найденное значение из нескольких возможных переменных окружения.

    Поддержка нескольких имен нужна для совместимости со старыми .env-файлами,
    которые могли использовать названия MODEL_PATH, INFERENCE_CONFIG_PATH
    и SYSTEM_METADATA_PATH.
    """
    if isinstance(names, str):
        names = (names,)

    for name in names:
        value = _clean_env_value(os.getenv(name))

        if value:
            return value

    return default


def _get_str(names: str | tuple[str, ...] | list[str], default: str = "") -> str:
    return _get_first_str(names, default)


def _get_bool(names: str | tuple[str, ...] | list[str], default: bool = False) -> bool:
    value = _get_first_str(names, "")

    if not value:
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "y", "on", "да"}:
        return True

    if normalized in {"0", "false", "no", "n", "off", "нет"}:
        return False

    return default


def _get_int(names: str | tuple[str, ...] | list[str], default: int) -> int:
    value = _get_first_str(names, "")

    if not value:
        return default

    try:
        return int(value.strip())
    except ValueError:
        return default


def _get_float(names: str | tuple[str, ...] | list[str], default: float) -> float:
    value = _get_first_str(names, "")

    if not value:
        return default

    try:
        return float(value.strip())
    except ValueError:
        return default


def _resolve_path(raw_value: str, base_dir: Path = BASE_DIR) -> Path:
    """
    Преобразует строку пути в абсолютный путь.

    Относительный путь считается относительно корня проекта.
    """
    path = Path(raw_value)

    if path.is_absolute():
        return path

    return (base_dir / path).resolve()


def _get_path(
    names: str | tuple[str, ...] | list[str],
    default: str,
    base_dir: Path = BASE_DIR,
) -> Path:
    raw_value = _get_first_str(names, default)

    return _resolve_path(raw_value, base_dir=base_dir)


def _get_artifact_path(
    names: str | tuple[str, ...] | list[str],
    default_filename: str,
    artifact_dir: Path,
) -> Path:
    """
    Возвращает путь к файлу артефакта.

    Если в .env указан только файл без каталога, он ищется внутри каталога артефактов.
    Если указан относительный путь с каталогами, он считается относительно корня проекта.
    """
    raw_value = _get_first_str(names, "")

    if not raw_value:
        return (artifact_dir / default_filename).resolve()

    path = Path(raw_value)

    if path.is_absolute():
        return path

    if path.parent == Path("."):
        return (artifact_dir / path).resolve()

    return (BASE_DIR / path).resolve()


def _get_list(names: str | tuple[str, ...] | list[str], default: str = "") -> list[str]:
    value = _get_first_str(names, default)

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def _mask_secret(value: str) -> str:
    if not value:
        return "не указан"

    if len(value) <= 8:
        return "***"

    return value[:4] + "***" + value[-4:]


@dataclass(frozen=True)
class RuntimeConfig:
    upload_folder: Path
    result_folder: Path
    preview_folder: Path
    report_folder: Path
    database_path: Path
    max_content_length_mb: int
    allowed_extensions: tuple[str, ...]

    @property
    def max_content_length_bytes(self) -> int:
        return self.max_content_length_mb * 1024 * 1024

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {
            extension.lower().lstrip(".")
            for extension in self.allowed_extensions
        }


@dataclass(frozen=True)
class SegmentationArtifactsConfig:
    artifact_dir: Path
    model_path: Path
    config_path: Path
    metadata_path: Path
    labels_path: Path
    default_mode: str
    threshold: float


@dataclass(frozen=True)
class RestorationArtifactsConfig:
    artifact_dir: Path
    model_path: Path
    config_path: Path
    text_report_config_path: Path
    final_summary_path: Path
    by_type_summary_path: Path
    example_report_path: Path
    default_mode: str
    default_degradation_type: str


@dataclass(frozen=True)
class LLMConfig:
    use_llm_explanation: bool
    proxyapi_api_key: str
    openai_base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout_seconds: int

    @property
    def masked_api_key(self) -> str:
        return _mask_secret(self.proxyapi_api_key)


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    env_path: Path
    app_name: str
    app_version: str
    flask_env: str
    secret_key: str
    debug: bool
    device: str
    runtime: RuntimeConfig
    segmentation: SegmentationArtifactsConfig
    restoration: RestorationArtifactsConfig
    llm: LLMConfig

    @property
    def is_development(self) -> bool:
        return self.flask_env.lower() == "development"


def _build_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        upload_folder=_get_path("UPLOAD_FOLDER", "runtime/uploads"),
        result_folder=_get_path("RESULT_FOLDER", "runtime/results"),
        preview_folder=_get_path("PREVIEW_FOLDER", "runtime/previews"),
        report_folder=_get_path("REPORT_FOLDER", "runtime/reports"),
        database_path=_get_path("DATABASE_PATH", "runtime/app_history.sqlite3"),
        max_content_length_mb=_get_int("MAX_CONTENT_LENGTH_MB", 15),
        allowed_extensions=tuple(
            _get_list("ALLOWED_EXTENSIONS", "jpg,jpeg,png,webp")
        ),
    )


def _build_segmentation_config() -> SegmentationArtifactsConfig:
    artifact_dir = _get_path(
        "SEGMENTATION_ARTIFACT_DIR",
        "artifacts/segmentation",
    )

    return SegmentationArtifactsConfig(
        artifact_dir=artifact_dir,
        model_path=_get_artifact_path(
            (
                "SEGMENTATION_MODEL_PATH",
                "SEGMENTATION_MODEL_FILE",
                "MODEL_PATH",
            ),
            "segmentation_improved_unet_resnet34_best.pt",
            artifact_dir,
        ),
        config_path=_get_artifact_path(
            (
                "SEGMENTATION_CONFIG_PATH",
                "INFERENCE_CONFIG_PATH",
            ),
            "segmentation_config.json",
            artifact_dir,
        ),
        metadata_path=_get_artifact_path(
            (
                "SEGMENTATION_METADATA_PATH",
                "SYSTEM_METADATA_PATH",
            ),
            "system_metadata.json",
            artifact_dir,
        ),
        labels_path=_get_artifact_path(
            "SEGMENTATION_LABELS_PATH",
            "segmentation_labels.json",
            artifact_dir,
        ),
        default_mode=_get_str(
            "SEGMENTATION_DEFAULT_MODE",
            "transparent_background",
        ),
        threshold=_get_float("SEGMENTATION_THRESHOLD", 0.5),
    )


def _build_restoration_config() -> RestorationArtifactsConfig:
    artifact_dir = _get_path(
        "RESTORATION_ARTIFACT_DIR",
        "artifacts/restoration",
    )

    return RestorationArtifactsConfig(
        artifact_dir=artifact_dir,
        model_path=_get_artifact_path(
            (
                "RESTORATION_MODEL_PATH",
                "RESTORATION_MODEL_FILE",
            ),
            "restoration_calibrated_model_second_experiment.pt",
            artifact_dir,
        ),
        config_path=_get_artifact_path(
            (
                "RESTORATION_CONFIG_PATH",
                "RESTORATION_INFERENCE_CONFIG_PATH",
            ),
            "restoration_inference_config_second_experiment.json",
            artifact_dir,
        ),
        text_report_config_path=_get_artifact_path(
            (
                "RESTORATION_TEXT_REPORT_CONFIG_PATH",
                "TEXT_REPORT_CONFIG_PATH",
            ),
            "text_report_config_second_experiment.json",
            artifact_dir,
        ),
        final_summary_path=_get_artifact_path(
            (
                "RESTORATION_FINAL_SUMMARY_PATH",
                "FINAL_TEST_SUMMARY_PATH",
            ),
            "final_test_summary_by_mode_second_experiment.csv",
            artifact_dir,
        ),
        by_type_summary_path=_get_artifact_path(
            (
                "RESTORATION_BY_TYPE_SUMMARY_PATH",
                "FINAL_TEST_BY_TYPE_SUMMARY_PATH",
            ),
            "final_test_adaptive_by_type_second_experiment.csv",
            artifact_dir,
        ),
        example_report_path=_get_artifact_path(
            (
                "RESTORATION_EXAMPLE_REPORT_PATH",
                "USER_RESTORATION_REPORT_PATH",
            ),
            "user_restoration_report_second_experiment.md",
            artifact_dir,
        ),
        default_mode=_get_str(
            "RESTORATION_DEFAULT_MODE",
            "global",
        ),
        default_degradation_type=_get_str(
            "RESTORATION_DEFAULT_DEGRADATION_TYPE",
            "auto",
        ),
    )


def _build_llm_config() -> LLMConfig:
    return LLMConfig(
        use_llm_explanation=_get_bool("USE_LLM_EXPLANATION", False),
        proxyapi_api_key=_get_str(
            (
                "PROXYAPI_API_KEY",
                "OPENAI_API_KEY",
            ),
            "",
        ),
        openai_base_url=_get_str(
            (
                "OPENAI_BASE_URL",
                "PROXYAPI_BASE_URL",
            ),
            "https://api.proxyapi.ru/openai/v1",
        ),
        model_name=_get_str(
            (
                "MODEL_NAME",
                "OPENAI_MODEL_NAME",
                "LLM_MODEL_NAME",
            ),
            "gpt-4o-mini",
        ),
        temperature=_get_float("LLM_TEMPERATURE", 0.0),
        max_tokens=_get_int("LLM_MAX_TOKENS", 450),
        timeout_seconds=_get_int("LLM_TIMEOUT_SECONDS", 60),
    )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    runtime_config = _build_runtime_config()

    config = AppConfig(
        base_dir=BASE_DIR,
        env_path=ENV_PATH,
        app_name=_get_str(
            "APP_NAME",
            "Интеллектуальный сервис генерации и обработки изображений",
        ),
        app_version=_get_str("APP_VERSION", "1.0.0"),
        flask_env=_get_str("FLASK_ENV", "development"),
        secret_key=_get_str("SECRET_KEY", "change_me_to_long_random_string"),
        debug=_get_bool("DEBUG", True),
        device=_get_str("DEVICE", "auto"),
        runtime=runtime_config,
        segmentation=_build_segmentation_config(),
        restoration=_build_restoration_config(),
        llm=_build_llm_config(),
    )

    ensure_runtime_directories(config)

    return config


def ensure_runtime_directories(config: AppConfig) -> None:
    """
    Создает рабочие каталоги приложения.

    Каталоги runtime нужны для пользовательских загрузок, результатов,
    предварительных изображений, отчетов и базы истории.
    """
    directories = [
        config.runtime.upload_folder,
        config.runtime.result_folder,
        config.runtime.preview_folder,
        config.runtime.report_folder,
        config.runtime.database_path.parent,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename: str, config: AppConfig | None = None) -> bool:
    """
    Проверяет расширение пользовательского файла.
    """
    if config is None:
        config = get_config()

    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in config.runtime.allowed_extensions_set


def _path_status(path: Path, required: bool) -> dict[str, Any]:
    exists = path.exists()

    return {
        "path": str(path),
        "exists": exists,
        "required": required,
        "size_bytes": path.stat().st_size if exists and path.is_file() else 0,
    }


def get_artifact_status(config: AppConfig | None = None) -> dict[str, Any]:
    """
    Возвращает сведения о наличии артефактов.

    segmentation_labels.json оставлен как необязательный файл, потому что
    бинарная сегментация животного может работать по segmentation_config.json
    и system_metadata.json.
    """
    if config is None:
        config = get_config()

    return {
        "segmentation_model": _path_status(config.segmentation.model_path, True),
        "segmentation_config": _path_status(config.segmentation.config_path, True),
        "segmentation_metadata": _path_status(config.segmentation.metadata_path, False),
        "segmentation_labels": _path_status(config.segmentation.labels_path, False),
        "restoration_model": _path_status(config.restoration.model_path, True),
        "restoration_config": _path_status(config.restoration.config_path, True),
        "restoration_text_report_config": _path_status(
            config.restoration.text_report_config_path,
            True,
        ),
        "restoration_final_summary": _path_status(
            config.restoration.final_summary_path,
            False,
        ),
        "restoration_by_type_summary": _path_status(
            config.restoration.by_type_summary_path,
            False,
        ),
        "restoration_example_report": _path_status(
            config.restoration.example_report_path,
            False,
        ),
    }


def validate_required_artifacts(config: AppConfig | None = None) -> list[str]:
    """
    Проверяет только обязательные файлы приложения.

    Файл segmentation_labels.json не является обязательным. В текущей системе
    используется бинарная сегментация: объект животного и фон. Поэтому отсутствие
    отдельного файла меток не должно блокировать запуск приложения.
    """
    if config is None:
        config = get_config()

    required_paths = {
        "модель сегментации": config.segmentation.model_path,
        "настройки сегментации": config.segmentation.config_path,
        "модель восстановления": config.restoration.model_path,
        "настройки восстановления": config.restoration.config_path,
        "настройки текстового отчета": config.restoration.text_report_config_path,
    }

    problems: list[str] = []

    for title, path in required_paths.items():
        if not path.exists():
            problems.append(f"Не найден файл: {title} -> {path}")
        elif path.is_file() and path.stat().st_size == 0:
            problems.append(f"Файл пустой: {title} -> {path}")

    return problems


def get_public_config_summary(config: AppConfig | None = None) -> dict[str, Any]:
    """
    Возвращает безопасную сводку настроек без раскрытия ключа доступа.
    """
    if config is None:
        config = get_config()

    return {
        "app_name": config.app_name,
        "app_version": config.app_version,
        "flask_env": config.flask_env,
        "debug": config.debug,
        "device": config.device,
        "base_dir": str(config.base_dir),
        "env_path": str(config.env_path),
        "upload_folder": str(config.runtime.upload_folder),
        "result_folder": str(config.runtime.result_folder),
        "preview_folder": str(config.runtime.preview_folder),
        "report_folder": str(config.runtime.report_folder),
        "database_path": str(config.runtime.database_path),
        "max_content_length_mb": config.runtime.max_content_length_mb,
        "allowed_extensions": list(config.runtime.allowed_extensions),
        "segmentation_artifact_dir": str(config.segmentation.artifact_dir),
        "segmentation_model_path": str(config.segmentation.model_path),
        "segmentation_config_path": str(config.segmentation.config_path),
        "segmentation_metadata_path": str(config.segmentation.metadata_path),
        "segmentation_labels_path": str(config.segmentation.labels_path),
        "segmentation_labels_required": False,
        "restoration_artifact_dir": str(config.restoration.artifact_dir),
        "restoration_model_path": str(config.restoration.model_path),
        "restoration_config_path": str(config.restoration.config_path),
        "restoration_text_report_config_path": str(
            config.restoration.text_report_config_path
        ),
        "use_llm_explanation": config.llm.use_llm_explanation,
        "openai_base_url": config.llm.openai_base_url,
        "model_name": config.llm.model_name,
        "masked_api_key": config.llm.masked_api_key,
    }


CONFIG = get_config()
APP_CONFIG = CONFIG
BASE_CONFIG = CONFIG
