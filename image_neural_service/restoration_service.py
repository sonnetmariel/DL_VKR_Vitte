# restoration_service.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from config import AppConfig, get_config
from image_utils import (
    ImageQualityFeatures,
    SavedImageInfo,
    create_side_by_side,
    describe_quality_features,
    estimate_image_quality_features,
    load_image,
    path_to_url,
    resize_back,
    resize_for_model,
    save_runtime_image,
)
from model_registry import RestorationModelBundle, get_restoration_bundle


class RestorationServiceError(RuntimeError):
    """
    Ошибка работы сервиса восстановления качества изображения.
    """


@dataclass(frozen=True)
class RestorationQualitySummary:
    """
    Простые показатели изображения без эталона.

    В пользовательском приложении обычно нет эталонного изображения, поэтому
    невозможно честно рассчитать показатели из эксперимента: структурное сходство,
    отношение сигнала к искажению и среднюю абсолютную ошибку относительно эталона.
    Вместо этого сохраняются простые признаки самого изображения.
    """

    before: dict[str, float]
    after: dict[str, float]
    difference: dict[str, float]
    before_text: dict[str, str]
    after_text: dict[str, str]


@dataclass(frozen=True)
class RestorationResult:
    """
    Результат восстановления изображения.
    """

    mode: str
    mode_name_ru: str
    degradation_type: str
    degradation_type_name_ru: str
    applied_correction_strength: float
    input_path: str
    input_url: str
    result_image: SavedImageInfo
    preview_image: SavedImageInfo
    quality_summary: RestorationQualitySummary
    model_info: dict[str, Any]
    experiment_reference: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "mode_name_ru": self.mode_name_ru,
            "degradation_type": self.degradation_type,
            "degradation_type_name_ru": self.degradation_type_name_ru,
            "applied_correction_strength": self.applied_correction_strength,
            "input_path": self.input_path,
            "input_url": self.input_url,
            "result_image": asdict(self.result_image),
            "preview_image": asdict(self.preview_image),
            "quality_summary": asdict(self.quality_summary),
            "model_info": self.model_info,
            "experiment_reference": self.experiment_reference,
            "warnings": self.warnings,
        }


RESTORATION_MODE_NAMES_RU = {
    "none": "без восстановления",
    "global": "калиброванная схема, общий режим",
    "adaptive": "калиброванная схема, адаптивный режим",
}


RESTORATION_MODE_ALIASES = {
    "none": "none",
    "off": "none",
    "without": "none",
    "без восстановления": "none",
    "global": "global",
    "common": "global",
    "auto": "global",
    "automatic": "global",
    "общий": "global",
    "автоматический": "global",
    "калиброванная схема, общий режим": "global",
    "adaptive": "adaptive",
    "type": "adaptive",
    "by_type": "adaptive",
    "адаптивный": "adaptive",
    "по типу ухудшения": "adaptive",
    "калиброванная схема, адаптивный режим": "adaptive",
}


DEGRADATION_TYPE_NAMES_RU = {
    "auto": "тип ухудшения не указан",
    "brightness_contrast": "изменение яркости и контрастности",
    "combined": "комбинированное ухудшение",
    "downscale_upscale": "уменьшение и последующее увеличение размера",
    "gaussian_blur": "гауссово размытие",
    "gaussian_noise": "гауссов шум",
    "jpeg_compression": "сжатие изображения",
    "motion_blur": "размытие движения",
}


DEGRADATION_TYPE_ALIASES = {
    "auto": "auto",
    "unknown": "auto",
    "не указано": "auto",
    "тип ухудшения не указан": "auto",
    "brightness": "brightness_contrast",
    "contrast": "brightness_contrast",
    "brightness_contrast": "brightness_contrast",
    "яркость": "brightness_contrast",
    "контрастность": "brightness_contrast",
    "изменение яркости и контрастности": "brightness_contrast",
    "combined": "combined",
    "mixed": "combined",
    "комбинированное": "combined",
    "комбинированное ухудшение": "combined",
    "downscale": "downscale_upscale",
    "upscale": "downscale_upscale",
    "downscale_upscale": "downscale_upscale",
    "уменьшение и увеличение": "downscale_upscale",
    "уменьшение и последующее увеличение размера": "downscale_upscale",
    "blur": "gaussian_blur",
    "gaussian_blur": "gaussian_blur",
    "gaussian blur": "gaussian_blur",
    "размытие": "gaussian_blur",
    "гауссово размытие": "gaussian_blur",
    "noise": "gaussian_noise",
    "gaussian_noise": "gaussian_noise",
    "gaussian noise": "gaussian_noise",
    "шум": "gaussian_noise",
    "гауссов шум": "gaussian_noise",
    "jpeg": "jpeg_compression",
    "jpg": "jpeg_compression",
    "compression": "jpeg_compression",
    "jpeg_compression": "jpeg_compression",
    "сжатие": "jpeg_compression",
    "сжатие изображения": "jpeg_compression",
    "motion": "motion_blur",
    "motion_blur": "motion_blur",
    "motion blur": "motion_blur",
    "размытие движения": "motion_blur",
}


def normalize_restoration_mode(mode: str | None, default: str = "global") -> str:
    """
    Приводит режим восстановления к внутреннему названию.
    """
    if not mode:
        return default

    normalized = str(mode).strip().lower()

    return RESTORATION_MODE_ALIASES.get(normalized, default)


def normalize_degradation_type(degradation_type: str | None, default: str = "auto") -> str:
    """
    Приводит тип ухудшения к внутреннему названию.
    """
    if not degradation_type:
        return default

    normalized = str(degradation_type).strip().lower()

    return DEGRADATION_TYPE_ALIASES.get(normalized, default)


def get_available_restoration_modes() -> list[dict[str, str]]:
    """
    Возвращает режимы восстановления для интерфейса.
    """
    return [
        {"value": key, "name_ru": value}
        for key, value in RESTORATION_MODE_NAMES_RU.items()
    ]


def get_available_degradation_types() -> list[dict[str, str]]:
    """
    Возвращает типы ухудшений для адаптивного режима.
    """
    return [
        {"value": key, "name_ru": value}
        for key, value in DEGRADATION_TYPE_NAMES_RU.items()
    ]


def tensor_from_image_for_restoration(
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """
    Переводит изображение в тензор для модели восстановления.

    Для модели восстановления используется диапазон значений от 0 до 1.
    """
    image_array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

    return tensor.to(device)


def image_from_tensor_for_restoration(tensor: torch.Tensor) -> Image.Image:
    """
    Переводит выход модели восстановления в изображение.
    """
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    array = tensor.squeeze(0).permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(array, mode="RGB")


def calculate_feature_difference(
    before: ImageQualityFeatures,
    after: ImageQualityFeatures,
) -> dict[str, float]:
    """
    Рассчитывает разницу простых признаков после обработки.
    """
    return {
        "brightness": round(after.brightness - before.brightness, 4),
        "contrast": round(after.contrast - before.contrast, 4),
        "sharpness": round(after.sharpness - before.sharpness, 4),
        "colorfulness": round(after.colorfulness - before.colorfulness, 4),
    }


def quality_features_to_dict(features: ImageQualityFeatures) -> dict[str, float]:
    """
    Переводит признаки изображения в словарь.
    """
    return {
        "width": float(features.width),
        "height": float(features.height),
        "brightness": round(float(features.brightness), 4),
        "contrast": round(float(features.contrast), 4),
        "sharpness": round(float(features.sharpness), 4),
        "colorfulness": round(float(features.colorfulness), 4),
    }


def build_quality_summary(
    before_image: Image.Image,
    after_image: Image.Image,
) -> RestorationQualitySummary:
    """
    Формирует простое описание изменений изображения без эталона.
    """
    before_features = estimate_image_quality_features(before_image)
    after_features = estimate_image_quality_features(after_image)

    return RestorationQualitySummary(
        before=quality_features_to_dict(before_features),
        after=quality_features_to_dict(after_features),
        difference=calculate_feature_difference(before_features, after_features),
        before_text=describe_quality_features(before_features),
        after_text=describe_quality_features(after_features),
    )


def load_csv_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """
    Загружает таблицу CSV как список словарей.

    Если pandas недоступен или файл отсутствует, возвращается пустой список.
    """
    if not path.exists():
        return []

    try:
        import pandas as pd

        table = pd.read_csv(path)

        if limit is not None:
            table = table.head(limit)

        return table.to_dict(orient="records")
    except Exception:
        return []


def build_restoration_model_info(bundle: RestorationModelBundle) -> dict[str, Any]:
    """
    Формирует сведения о модели восстановления для результата обработки.
    """
    return {
        "model_name": bundle.config.get(
            "model_name",
            "restoration_first_residual_model",
        ),
        "image_size": bundle.image_size,
        "global_correction_strength": bundle.global_correction_strength,
        "correction_strength_by_type": bundle.correction_strength_by_type,
        "model_path": str(bundle.model_path),
        "device": str(bundle.device),
        "checkpoint_keys": bundle.checkpoint_keys,
        "load_strict": bundle.load_report.strict,
        "missing_keys": bundle.load_report.missing_keys[:10],
        "unexpected_keys": bundle.load_report.unexpected_keys[:10],
    }


class RestorationService:
    """
    Сервис восстановления качества изображения.

    Сервис использует модель второго эксперимента. Модель рассчитывает поправку
    к изображению, а калибровочный коэффициент управляет силой этой поправки.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        bundle: RestorationModelBundle | None = None,
    ) -> None:
        self.config = config or get_config()
        self.bundle = bundle or get_restoration_bundle()

    def get_correction_strength(
        self,
        mode: str,
        degradation_type: str,
    ) -> tuple[float, list[str]]:
        """
        Возвращает коэффициент силы нейросетевой правки.

        Для общего режима используется общий коэффициент.
        Для адаптивного режима используется коэффициент по типу ухудшения.
        Если тип ухудшения неизвестен, применяется общий режим.
        """
        warnings: list[str] = []

        normalized_mode = normalize_restoration_mode(
            mode,
            default=self.config.restoration.default_mode,
        )

        normalized_type = normalize_degradation_type(
            degradation_type,
            default=self.config.restoration.default_degradation_type,
        )

        if normalized_mode == "none":
            return 0.0, warnings

        if normalized_mode == "global":
            return float(self.bundle.global_correction_strength), warnings

        if normalized_mode == "adaptive":
            if normalized_type == "auto":
                warnings.append(
                    "Тип ухудшения не указан. Для восстановления применен общий коэффициент."
                )
                return float(self.bundle.global_correction_strength), warnings

            if normalized_type not in self.bundle.correction_strength_by_type:
                warnings.append(
                    "Для выбранного типа ухудшения нет отдельного коэффициента. "
                    "Для восстановления применен общий коэффициент."
                )
                return float(self.bundle.global_correction_strength), warnings

            return float(self.bundle.correction_strength_by_type[normalized_type]), warnings

        return float(self.bundle.global_correction_strength), warnings

    def restore_pil_image(
        self,
        image: Image.Image,
        mode: str | None = None,
        degradation_type: str | None = None,
    ) -> tuple[Image.Image, float, list[str]]:
        """
        Восстанавливает изображение без сохранения файлов.

        Возвращает восстановленное изображение, примененный коэффициент и предупреждения.
        """
        normalized_mode = normalize_restoration_mode(
            mode,
            default=self.config.restoration.default_mode,
        )

        normalized_type = normalize_degradation_type(
            degradation_type,
            default=self.config.restoration.default_degradation_type,
        )

        correction_strength, warnings = self.get_correction_strength(
            mode=normalized_mode,
            degradation_type=normalized_type,
        )

        original_image = image.convert("RGB")

        if correction_strength == 0.0:
            warnings.append(
                "Коэффициент восстановления равен нулю. Изображение оставлено без нейросетевой правки."
            )
            return original_image.copy(), correction_strength, warnings

        prepared_image, original_size = resize_for_model(
            image=original_image,
            image_size=self.bundle.image_size,
            keep_aspect_ratio=False,
        )

        input_tensor = tensor_from_image_for_restoration(
            image=prepared_image,
            device=self.bundle.device,
        )

        self.bundle.model.eval()

        with torch.no_grad():
            base_restored_tensor = self.bundle.model(input_tensor)

        # В эксперименте калибровка применялась как изменение силы поправки:
        # итог = вход + коэффициент * (выход модели - вход).
        calibrated_tensor = input_tensor + correction_strength * (
            base_restored_tensor - input_tensor
        )

        calibrated_tensor = torch.clamp(calibrated_tensor, 0.0, 1.0)

        restored_model_size = image_from_tensor_for_restoration(calibrated_tensor)
        restored_original_size = resize_back(restored_model_size, original_size)

        return restored_original_size, correction_strength, warnings

    def process_image(
        self,
        input_path: str | Path,
        mode: str | None = None,
        degradation_type: str | None = None,
    ) -> RestorationResult:
        """
        Выполняет восстановление качества изображения и сохраняет файлы результата.
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise RestorationServiceError(f"Исходное изображение не найдено: {input_path}")

        normalized_mode = normalize_restoration_mode(
            mode,
            default=self.config.restoration.default_mode,
        )

        normalized_type = normalize_degradation_type(
            degradation_type,
            default=self.config.restoration.default_degradation_type,
        )

        source_filename = input_path.name
        original_image = load_image(input_path)

        restored_image, correction_strength, warnings = self.restore_pil_image(
            image=original_image,
            mode=normalized_mode,
            degradation_type=normalized_type,
        )

        result_info = save_runtime_image(
            image=restored_image,
            folder=self.config.runtime.result_folder,
            source_filename=source_filename,
            suffix=f"restoration_{normalized_mode}_{normalized_type}",
            extension="png",
            config=self.config,
        )

        preview = create_side_by_side(
            left_image=original_image,
            right_image=restored_image,
            left_title="До восстановления",
            right_title="После восстановления",
        )

        preview_info = save_runtime_image(
            image=preview,
            folder=self.config.runtime.preview_folder,
            source_filename=source_filename,
            suffix=f"preview_restoration_{normalized_mode}_{normalized_type}",
            extension="png",
            config=self.config,
        )

        quality_summary = build_quality_summary(
            before_image=original_image,
            after_image=restored_image,
        )

        warnings.extend(
            self._build_warnings(
                mode=normalized_mode,
                degradation_type=normalized_type,
                correction_strength=correction_strength,
                quality_summary=quality_summary,
            )
        )

        return RestorationResult(
            mode=normalized_mode,
            mode_name_ru=RESTORATION_MODE_NAMES_RU.get(normalized_mode, normalized_mode),
            degradation_type=normalized_type,
            degradation_type_name_ru=DEGRADATION_TYPE_NAMES_RU.get(
                normalized_type,
                normalized_type,
            ),
            applied_correction_strength=correction_strength,
            input_path=str(input_path),
            input_url=path_to_url(input_path, self.config),
            result_image=result_info,
            preview_image=preview_info,
            quality_summary=quality_summary,
            model_info=build_restoration_model_info(self.bundle),
            experiment_reference=self.get_experiment_reference(),
            warnings=warnings,
        )

    def get_experiment_reference(self) -> dict[str, Any]:
        """
        Возвращает сведения о результатах второго эксперимента.

        Эти данные используются в пользовательском отчете и на странице о системе.
        """
        final_summary = load_csv_records(
            self.config.restoration.final_summary_path,
            limit=None,
        )

        by_type_summary = load_csv_records(
            self.config.restoration.by_type_summary_path,
            limit=None,
        )

        best_mode = None

        for record in final_summary:
            if record.get("mode_name") == "калиброванная схема, адаптивный режим":
                best_mode = record
                break

        if best_mode is None and final_summary:
            best_mode = final_summary[0]

        return {
            "final_summary": final_summary,
            "adaptive_by_type_summary": by_type_summary,
            "best_mode": best_mode,
            "important_note": (
                "В пользовательском приложении нет эталонного изображения, поэтому "
                "экспериментальные показатели структурного сходства и отношения сигнала "
                "к искажению приводятся как справочная информация по контрольной части."
            ),
        }

    def _build_warnings(
        self,
        mode: str,
        degradation_type: str,
        correction_strength: float,
        quality_summary: RestorationQualitySummary,
    ) -> list[str]:
        """
        Формирует предупреждения для пользователя.
        """
        warnings: list[str] = []

        if mode == "adaptive" and degradation_type == "brightness_contrast":
            warnings.append(
                "Для изменения яркости и контрастности калибровка второго эксперимента "
                "показала нулевую пользу нейросетевой правки."
            )

        if correction_strength > 1.0:
            warnings.append(
                "Использован усиленный коэффициент восстановления. Результат нужно "
                "проверить визуально, так как возможны небольшие искусственные изменения деталей."
            )

        sharpness_change = quality_summary.difference.get("sharpness", 0.0)

        if sharpness_change < -2.0:
            warnings.append(
                "После восстановления простая оценка резкости снизилась. "
                "Возможно, изображение стало более сглаженным."
            )

        return warnings

    def get_model_card(self) -> dict[str, Any]:
        """
        Возвращает карточку модели восстановления для страницы о системе.
        """
        return {
            "title": "Модель восстановления качества изображения",
            "description": (
                "Модель рассчитывает нейросетевую поправку к изображению и применяет "
                "калиброванный коэффициент восстановления."
            ),
            "model_info": build_restoration_model_info(self.bundle),
            "available_modes": get_available_restoration_modes(),
            "available_degradation_types": get_available_degradation_types(),
            "experiment_reference": self.get_experiment_reference(),
        }


def get_restoration_service() -> RestorationService:
    """
    Создает сервис восстановления с кэшированной моделью.
    """
    return RestorationService()


def run_restoration(
    input_path: str | Path,
    mode: str | None = None,
    degradation_type: str | None = None,
) -> RestorationResult:
    """
    Удобная функция для вызова восстановления из app.py.
    """
    service = get_restoration_service()

    return service.process_image(
        input_path=input_path,
        mode=mode,
        degradation_type=degradation_type,
    )
