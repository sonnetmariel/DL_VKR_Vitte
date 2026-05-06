# model_registry.py
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from config import AppConfig, get_config, validate_required_artifacts


class ModelRegistryError(RuntimeError):
    """
    Ошибка загрузки или подготовки моделей приложения.
    """


@dataclass
class StateDictLoadReport:
    """
    Отчет о загрузке весов модели.

    missing_keys показывает параметры, которые были в модели, но не найдены в файле.
    unexpected_keys показывает параметры, которые были в файле, но не подошли к модели.
    """

    strict: bool
    missing_keys: list[str]
    unexpected_keys: list[str]


@dataclass
class SegmentationModelBundle:
    """
    Полный комплект сегментационной модели.

    Модель выделяет область животного и применяется для маски, прозрачного фона,
    размытия фона и замены фона.
    """

    model: nn.Module
    device: torch.device
    image_size: int
    image_mean: list[float]
    image_std: list[float]
    threshold: float
    config: dict[str, Any]
    metadata: dict[str, Any]
    labels: dict[str, Any]
    model_path: Path
    load_report: StateDictLoadReport


@dataclass
class RestorationModelBundle:
    """
    Полный комплект модели восстановления качества.

    Модель рассчитывает нейросетевую поправку к ухудшенному изображению.
    Калибровочные коэффициенты управляют силой применения этой поправки.
    """

    model: nn.Module
    device: torch.device
    image_size: int
    global_correction_strength: float
    correction_strength_by_type: dict[str, float]
    config: dict[str, Any]
    text_report_config: dict[str, Any]
    model_path: Path
    checkpoint_keys: list[str]
    load_report: StateDictLoadReport


class ResidualConvBlock(nn.Module):
    """
    Остаточный сверточный блок для модели восстановления.

    Остаточная связь помогает сохранять структуру исходного изображения.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor + self.block(input_tensor)


class FirstResidualRestorationModel(nn.Module):
    """
    Остаточная модель восстановления качества изображений.

    Модель не создает изображение с нуля, а рассчитывает поправку к входному
    изображению. Это снижает риск потери структуры и цвета.
    """

    def __init__(
        self,
        channels: int = 64,
        block_count: int = 8,
        correction_strength: float = 0.35,
    ) -> None:
        super().__init__()

        self.correction_strength = correction_strength

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[ResidualConvBlock(channels) for _ in range(block_count)]
        )

        self.output_layer = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        features = self.input_layer(input_tensor)
        features = self.blocks(features)

        correction = torch.tanh(self.output_layer(features))
        restored = input_tensor + self.correction_strength * correction
        restored = torch.clamp(restored, 0.0, 1.0)

        return restored


def read_json_file(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Читает файл JSON.

    Если файл отсутствует, возвращается значение по умолчанию.
    """
    if default is None:
        default = {}

    if not path.exists():
        return default

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

        return default
    except Exception as exc:
        raise ModelRegistryError(f"Не удалось прочитать JSON-файл: {path}") from exc


def resolve_device(device_name: str = "auto") -> torch.device:
    """
    Выбирает устройство вычислений.

    auto означает использование графического ускорителя при наличии.
    """
    normalized = (device_name or "auto").strip().lower()

    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if normalized == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(normalized)


def load_torch_artifact(path: Path, device: torch.device) -> Any:
    """
    Загружает файл PyTorch.

    weights_only отключается, потому что экспериментальные файлы могут содержать
    не только веса, но и калибровочные коэффициенты.
    """
    if not path.exists():
        raise ModelRegistryError(f"Не найден файл модели: {path}")

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception as exc:
        raise ModelRegistryError(f"Не удалось загрузить файл модели: {path}") from exc


def looks_like_state_dict(value: Any) -> bool:
    """
    Проверяет, похож ли объект на словарь весов модели.
    """
    if not isinstance(value, (dict, OrderedDict)):
        return False

    if not value:
        return False

    tensor_count = 0

    for item in value.values():
        if torch.is_tensor(item):
            tensor_count += 1

    return tensor_count > 0


def extract_state_dict(artifact: Any) -> OrderedDict[str, torch.Tensor]:
    """
    Извлекает словарь весов из разных возможных форматов чекпоинта.
    """
    if isinstance(artifact, nn.Module):
        return OrderedDict(artifact.state_dict())

    if looks_like_state_dict(artifact):
        return OrderedDict(artifact)

    if isinstance(artifact, dict):
        candidate_keys = [
            "model_state_dict",
            "state_dict",
            "model",
            "net",
            "network",
            "weights",
        ]

        for key in candidate_keys:
            value = artifact.get(key)

            if isinstance(value, nn.Module):
                return OrderedDict(value.state_dict())

            if looks_like_state_dict(value):
                return OrderedDict(value)

    raise ModelRegistryError(
        "Не удалось найти веса модели в файле. "
        "Ожидался state_dict или словарь с ключом model_state_dict."
    )


def clean_state_dict_keys(
    state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """
    Убирает распространенные префиксы из имен параметров.

    Это нужно, если модель сохранялась через DataParallel или внутри обертки.
    """
    cleaned: OrderedDict[str, torch.Tensor] = OrderedDict()

    prefixes = [
        "module.",
        "model.",
        "net.",
        "network.",
    ]

    for key, value in state_dict.items():
        new_key = key

        changed = True
        while changed:
            changed = False

            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def load_state_dict_safely(
    model: nn.Module,
    artifact: Any,
) -> StateDictLoadReport:
    """
    Загружает веса в модель.

    Сначала выполняется строгая загрузка. Если она не сработала, применяется
    нестрогая загрузка, чтобы приложение могло работать с чекпоинтами,
    содержащими дополнительные служебные параметры.
    """
    state_dict = extract_state_dict(artifact)
    state_dict = clean_state_dict_keys(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)

        return StateDictLoadReport(
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
        )
    except RuntimeError:
        load_result = model.load_state_dict(state_dict, strict=False)

        return StateDictLoadReport(
            strict=False,
            missing_keys=list(load_result.missing_keys),
            unexpected_keys=list(load_result.unexpected_keys),
        )


def create_segmentation_model(segmentation_config: dict[str, Any]) -> nn.Module:
    """
    Создает модель сегментации по настройкам первого эксперимента.

    В эксперименте использовалась архитектура Unet с кодировщиком resnet34.
    """
    architecture = str(segmentation_config.get("architecture", "Unet")).lower()
    encoder_name = str(segmentation_config.get("encoder_name", "resnet34"))

    if architecture not in {"unet", "u-net"}:
        raise ModelRegistryError(
            "Поддерживается только архитектура Unet. "
            f"В файле настроек указано: {segmentation_config.get('architecture')}"
        )

    try:
        import segmentation_models_pytorch as smp
    except Exception as exc:
        raise ModelRegistryError(
            "Не установлена библиотека segmentation_models_pytorch. "
            "Добавьте ее в requirements.txt и выполните установку зависимостей."
        ) from exc

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )

    return model


def get_int_from_sources(
    sources: list[dict[str, Any]],
    keys: list[str],
    default: int,
) -> int:
    """
    Извлекает целочисленное значение из нескольких словарей.
    """
    for source in sources:
        for key in keys:
            value = source.get(key)

            if value is None:
                continue

            try:
                return int(value)
            except (TypeError, ValueError):
                continue

    return default


def get_float_from_sources(
    sources: list[dict[str, Any]],
    keys: list[str],
    default: float,
) -> float:
    """
    Извлекает вещественное значение из нескольких словарей.
    """
    for source in sources:
        for key in keys:
            value = source.get(key)

            if value is None:
                continue

            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    return default


def get_list_from_sources(
    sources: list[dict[str, Any]],
    keys: list[str],
    default: list[float],
) -> list[float]:
    """
    Извлекает список чисел из нескольких словарей.
    """
    for source in sources:
        for key in keys:
            value = source.get(key)

            if isinstance(value, list) and value:
                try:
                    return [float(item) for item in value]
                except (TypeError, ValueError):
                    continue

    return default


def extract_checkpoint_keys(artifact: Any) -> list[str]:
    """
    Возвращает верхнеуровневые ключи чекпоинта для диагностики.
    """
    if isinstance(artifact, dict):
        return sorted(str(key) for key in artifact.keys())

    return [type(artifact).__name__]


def extract_correction_strength_by_type(
    artifact: Any,
    restoration_config: dict[str, Any],
) -> dict[str, float]:
    """
    Извлекает коэффициенты восстановления по типам ухудшений.
    """
    default_values = {
        "brightness_contrast": 0.0,
        "combined": 1.0,
        "downscale_upscale": 1.2,
        "gaussian_blur": 1.3,
        "gaussian_noise": 1.0,
        "jpeg_compression": 1.0,
        "motion_blur": 1.2,
    }

    candidates: list[Any] = []

    if isinstance(artifact, dict):
        candidates.extend([
            artifact.get("correction_strength_by_type"),
            artifact.get("best_correction_strength_by_type"),
            artifact.get("strength_by_type"),
            artifact.get("коэффициенты_по_типам"),
        ])

    candidates.extend([
        restoration_config.get("correction_strength_by_type"),
        restoration_config.get("best_correction_strength_by_type"),
        restoration_config.get("strength_by_type"),
        restoration_config.get("коэффициенты_по_типам"),
    ])

    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            result: dict[str, float] = {}

            for key, value in candidate.items():
                try:
                    result[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

            if result:
                return result

    return default_values


def extract_global_correction_strength(
    artifact: Any,
    restoration_config: dict[str, Any],
) -> float:
    """
    Извлекает общий коэффициент восстановления.
    """
    candidate_values: list[Any] = []

    if isinstance(artifact, dict):
        candidate_values.extend([
            artifact.get("best_global_correction_strength"),
            artifact.get("global_correction_strength"),
            artifact.get("correction_strength"),
            artifact.get("общий_коэффициент"),
        ])

    candidate_values.extend([
        restoration_config.get("best_global_correction_strength"),
        restoration_config.get("global_correction_strength"),
        restoration_config.get("correction_strength"),
        restoration_config.get("общий_коэффициент"),
    ])

    for value in candidate_values:
        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return 1.0


@lru_cache(maxsize=1)
def load_segmentation_bundle() -> SegmentationModelBundle:
    """
    Загружает и кэширует модель сегментации первого эксперимента.
    """
    config = get_config()
    device = resolve_device(config.device)

    segmentation_config = read_json_file(config.segmentation.config_path)
    segmentation_metadata = read_json_file(config.segmentation.metadata_path)
    segmentation_labels = read_json_file(config.segmentation.labels_path)

    model = create_segmentation_model(segmentation_config)
    artifact = load_torch_artifact(config.segmentation.model_path, device)

    load_report = load_state_dict_safely(model, artifact)

    model.to(device)
    model.eval()

    image_size = get_int_from_sources(
        [segmentation_config, segmentation_metadata],
        ["image_size", "input_size", "рабочий_размер"],
        256,
    )

    threshold = get_float_from_sources(
        [segmentation_config, segmentation_metadata, segmentation_labels],
        ["threshold", "порог"],
        config.segmentation.threshold,
    )

    image_mean = get_list_from_sources(
        [segmentation_config],
        ["image_mean", "mean"],
        [0.485, 0.456, 0.406],
    )

    image_std = get_list_from_sources(
        [segmentation_config],
        ["image_std", "std"],
        [0.229, 0.224, 0.225],
    )

    return SegmentationModelBundle(
        model=model,
        device=device,
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        threshold=threshold,
        config=segmentation_config,
        metadata=segmentation_metadata,
        labels=segmentation_labels,
        model_path=config.segmentation.model_path,
        load_report=load_report,
    )


@lru_cache(maxsize=1)
def load_restoration_bundle() -> RestorationModelBundle:
    """
    Загружает и кэширует модель восстановления второго эксперимента.
    """
    config = get_config()
    device = resolve_device(config.device)

    restoration_config = read_json_file(config.restoration.config_path)
    text_report_config = read_json_file(config.restoration.text_report_config_path)

    artifact = load_torch_artifact(config.restoration.model_path, device)

    model = FirstResidualRestorationModel(
        channels=64,
        block_count=8,
        correction_strength=0.35,
    )

    load_report = load_state_dict_safely(model, artifact)

    model.to(device)
    model.eval()

    image_size = get_int_from_sources(
        [restoration_config],
        [
            "image_size",
            "working_image_size",
            "рабочий_размер_изображения",
            "target_size",
        ],
        256,
    )

    global_correction_strength = extract_global_correction_strength(
        artifact=artifact,
        restoration_config=restoration_config,
    )

    correction_strength_by_type = extract_correction_strength_by_type(
        artifact=artifact,
        restoration_config=restoration_config,
    )

    return RestorationModelBundle(
        model=model,
        device=device,
        image_size=image_size,
        global_correction_strength=global_correction_strength,
        correction_strength_by_type=correction_strength_by_type,
        config=restoration_config,
        text_report_config=text_report_config,
        model_path=config.restoration.model_path,
        checkpoint_keys=extract_checkpoint_keys(artifact),
        load_report=load_report,
    )


class ModelRegistry:
    """
    Единый реестр моделей приложения.

    Остальные модули обращаются к этому классу и не работают напрямую с путями
    к артефактам.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    def get_device(self) -> torch.device:
        return resolve_device(self.config.device)

    def get_segmentation(self) -> SegmentationModelBundle:
        return load_segmentation_bundle()

    def get_restoration(self) -> RestorationModelBundle:
        return load_restoration_bundle()

    def warmup(self) -> dict[str, Any]:
        """
        Загружает обе модели и выполняет небольшой пробный прогон.

        Метод полезен при старте приложения, чтобы первая пользовательская обработка
        не была слишком медленной.
        """
        problems = validate_required_artifacts(self.config)

        if problems:
            return {
                "success": False,
                "problems": problems,
                "segmentation_loaded": False,
                "restoration_loaded": False,
            }

        segmentation_loaded = False
        restoration_loaded = False
        warmup_errors: list[str] = []

        try:
            segmentation = self.get_segmentation()
            dummy_segmentation = torch.zeros(
                1,
                3,
                segmentation.image_size,
                segmentation.image_size,
                device=segmentation.device,
            )

            with torch.no_grad():
                _ = segmentation.model(dummy_segmentation)

            segmentation_loaded = True
        except Exception as exc:
            warmup_errors.append(f"Сегментация: {exc}")

        try:
            restoration = self.get_restoration()
            dummy_restoration = torch.zeros(
                1,
                3,
                restoration.image_size,
                restoration.image_size,
                device=restoration.device,
            )

            with torch.no_grad():
                _ = restoration.model(dummy_restoration)

            restoration_loaded = True
        except Exception as exc:
            warmup_errors.append(f"Восстановление: {exc}")

        return {
            "success": segmentation_loaded and restoration_loaded and not warmup_errors,
            "problems": warmup_errors,
            "segmentation_loaded": segmentation_loaded,
            "restoration_loaded": restoration_loaded,
            "device": str(self.get_device()),
        }

    def status(self) -> dict[str, Any]:
        """
        Возвращает диагностическую информацию о моделях и артефактах.
        """
        problems = validate_required_artifacts(self.config)

        status: dict[str, Any] = {
            "required_artifact_problems": problems,
            "device": str(self.get_device()),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "",
            "segmentation": {
                "model_path": str(self.config.segmentation.model_path),
                "config_path": str(self.config.segmentation.config_path),
                "metadata_path": str(self.config.segmentation.metadata_path),
                "labels_path": str(self.config.segmentation.labels_path),
            },
            "restoration": {
                "model_path": str(self.config.restoration.model_path),
                "config_path": str(self.config.restoration.config_path),
                "text_report_config_path": str(
                    self.config.restoration.text_report_config_path
                ),
            },
        }

        return status

    def clear_cache(self) -> None:
        """
        Очищает кэш моделей.

        Используется редко, например после замены файлов артефактов без перезапуска
        процесса.
        """
        load_segmentation_bundle.cache_clear()
        load_restoration_bundle.cache_clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    """
    Возвращает общий реестр моделей.
    """
    return ModelRegistry(get_config())


def get_segmentation_bundle() -> SegmentationModelBundle:
    """
    Быстрый доступ к сегментационной модели.
    """
    return get_model_registry().get_segmentation()


def get_restoration_bundle() -> RestorationModelBundle:
    """
    Быстрый доступ к модели восстановления.
    """
    return get_model_registry().get_restoration()
