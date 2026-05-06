# segmentation_service.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from config import AppConfig, get_config
from image_utils import (
    SavedImageInfo,
    apply_blur_background,
    apply_generated_background,
    apply_transparent_background,
    create_mask_preview,
    create_side_by_side,
    harden_mask,
    highlight_object,
    load_image,
    path_to_url,
    refine_mask,
    resize_back,
    resize_for_model,
    resize_mask_back,
    save_runtime_image,
)
from model_registry import SegmentationModelBundle, get_segmentation_bundle


class SegmentationServiceError(RuntimeError):
    """
    Ошибка работы сервиса сегментации.
    """


@dataclass(frozen=True)
class SegmentationMaskStats:
    """
    Численные сведения о найденной области объекта.
    """

    foreground_pixel_count: int
    total_pixel_count: int
    foreground_percent: float
    bbox_left: int | None
    bbox_top: int | None
    bbox_right: int | None
    bbox_bottom: int | None
    bbox_width: int | None
    bbox_height: int | None


@dataclass(frozen=True)
class SegmentationResult:
    """
    Результат сегментации и обработки фона.
    """

    mode: str
    mode_name_ru: str
    input_path: str
    input_url: str
    result_image: SavedImageInfo
    mask_image: SavedImageInfo
    preview_image: SavedImageInfo
    mask_stats: SegmentationMaskStats
    model_info: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "mode_name_ru": self.mode_name_ru,
            "input_path": self.input_path,
            "input_url": self.input_url,
            "result_image": asdict(self.result_image),
            "mask_image": asdict(self.mask_image),
            "preview_image": asdict(self.preview_image),
            "mask_stats": asdict(self.mask_stats),
            "model_info": self.model_info,
            "warnings": self.warnings,
        }


SEGMENTATION_MODE_NAMES_RU = {
    "binary_mask": "получение бинарной маски",
    "transparent_background": "выделение объекта на прозрачном фоне",
    "blur_background": "размытие фона",
    "replace_background": "замена фона на простой сгенерированный фон",
    "highlight_object": "выделение объекта цветом",
}


SEGMENTATION_MODE_ALIASES = {
    "mask": "binary_mask",
    "binary": "binary_mask",
    "binary_mask": "binary_mask",
    "получение бинарной маски": "binary_mask",
    "remove_background": "transparent_background",
    "transparent": "transparent_background",
    "transparent_background": "transparent_background",
    "выделение объекта на прозрачном фоне": "transparent_background",
    "blur": "blur_background",
    "blur_background": "blur_background",
    "размытие фона": "blur_background",
    "replace": "replace_background",
    "replace_background": "replace_background",
    "generated_background": "replace_background",
    "замена фона на сгенерированный простой фон": "replace_background",
    "замена фона на простой сгенерированный фон": "replace_background",
    "highlight": "highlight_object",
    "highlight_object": "highlight_object",
    "выделение объекта": "highlight_object",
}


BACKGROUND_VARIANTS_RU = {
    "soft_blue": "мягкий голубой фон",
    "warm_light": "теплый светлый фон",
    "gray_studio": "серый студийный фон",
    "green_soft": "мягкий зеленый фон",
}


def normalize_segmentation_mode(mode: str | None, default: str = "transparent_background") -> str:
    """
    Приводит режим сегментации к внутреннему названию.
    """
    if not mode:
        return default

    normalized = str(mode).strip().lower()

    return SEGMENTATION_MODE_ALIASES.get(normalized, default)


def get_available_segmentation_modes() -> list[dict[str, str]]:
    """
    Возвращает список режимов для интерфейса приложения.
    """
    return [
        {"value": key, "name_ru": value}
        for key, value in SEGMENTATION_MODE_NAMES_RU.items()
    ]


def get_available_background_variants() -> list[dict[str, str]]:
    """
    Возвращает список простых фонов для замены фона.
    """
    return [
        {"value": key, "name_ru": value}
        for key, value in BACKGROUND_VARIANTS_RU.items()
    ]


def tensor_from_image(
    image: Image.Image,
    image_mean: list[float],
    image_std: list[float],
    device: torch.device,
) -> torch.Tensor:
    """
    Переводит изображение в тензор для модели сегментации.

    Нормализация должна совпадать с настройками, использованными при обучении.
    """
    image_array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0

    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

    mean = torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1)

    tensor = (tensor - mean) / std

    return tensor.to(device)


def probability_map_from_output(output: Any) -> np.ndarray:
    """
    Извлекает карту вероятностей из выхода модели.

    Модель сегментации возвращает логиты. Сигмоида переводит их в вероятности
    принадлежности пикселя к объекту.
    """
    if isinstance(output, (tuple, list)):
        output = output[0]

    if isinstance(output, dict):
        for key in ["out", "mask", "masks", "logits", "prediction"]:
            if key in output:
                output = output[key]
                break

    if not torch.is_tensor(output):
        raise SegmentationServiceError("Модель вернула неподдерживаемый формат результата.")

    probabilities = torch.sigmoid(output)

    probability_map = probabilities.detach().float().cpu().numpy()

    if probability_map.ndim == 4:
        probability_map = probability_map[0, 0]

    elif probability_map.ndim == 3:
        probability_map = probability_map[0]

    return probability_map


def create_mask_from_probability(
    probability_map: np.ndarray,
    threshold: float,
    postprocess: bool = True,
) -> Image.Image:
    """
    Создает маску объекта из карты вероятностей.
    """
    binary_array = (probability_map >= threshold).astype(np.uint8) * 255
    mask = Image.fromarray(binary_array, mode="L")

    if postprocess:
        mask = refine_mask(mask)

    else:
        mask = harden_mask(mask)

    return mask


def calculate_mask_stats(mask: Image.Image) -> SegmentationMaskStats:
    """
    Рассчитывает площадь найденного объекта и ограничивающий прямоугольник.
    """
    mask_array = np.array(mask.convert("L"))
    foreground = mask_array > 0

    foreground_pixel_count = int(foreground.sum())
    total_pixel_count = int(mask_array.size)

    if total_pixel_count == 0:
        foreground_percent = 0.0
    else:
        foreground_percent = round(foreground_pixel_count / total_pixel_count * 100.0, 4)

    if foreground_pixel_count == 0:
        return SegmentationMaskStats(
            foreground_pixel_count=foreground_pixel_count,
            total_pixel_count=total_pixel_count,
            foreground_percent=foreground_percent,
            bbox_left=None,
            bbox_top=None,
            bbox_right=None,
            bbox_bottom=None,
            bbox_width=None,
            bbox_height=None,
        )

    y_indices, x_indices = np.where(foreground)

    left = int(x_indices.min())
    right = int(x_indices.max())
    top = int(y_indices.min())
    bottom = int(y_indices.max())

    return SegmentationMaskStats(
        foreground_pixel_count=foreground_pixel_count,
        total_pixel_count=total_pixel_count,
        foreground_percent=foreground_percent,
        bbox_left=left,
        bbox_top=top,
        bbox_right=right,
        bbox_bottom=bottom,
        bbox_width=right - left + 1,
        bbox_height=bottom - top + 1,
    )


def rgba_to_preview_rgb(image: Image.Image) -> Image.Image:
    """
    Делает изображение с прозрачностью пригодным для предпросмотра.

    Прозрачные области показываются светлым фоном.
    """
    if image.mode != "RGBA":
        return image.convert("RGB")

    background = Image.new("RGB", image.size, (246, 248, 251))
    background.paste(image, mask=image.getchannel("A"))

    return background


def build_segmentation_model_info(bundle: SegmentationModelBundle) -> dict[str, Any]:
    """
    Формирует краткие сведения о модели для результата обработки.
    """
    metadata = bundle.metadata or {}

    return {
        "model_name": metadata.get(
            "model_name",
            bundle.config.get("model_name", "segmentation_model"),
        ),
        "architecture": metadata.get(
            "architecture",
            bundle.config.get("architecture", "Unet"),
        ),
        "encoder": metadata.get(
            "encoder",
            bundle.config.get("encoder_name", "resnet34"),
        ),
        "image_size": bundle.image_size,
        "threshold": bundle.threshold,
        "control_dice": metadata.get("control_dice"),
        "control_iou": metadata.get("control_iou"),
        "control_pixel_accuracy": metadata.get("control_pixel_accuracy"),
        "model_path": str(bundle.model_path),
        "device": str(bundle.device),
        "load_strict": bundle.load_report.strict,
        "missing_keys": bundle.load_report.missing_keys[:10],
        "unexpected_keys": bundle.load_report.unexpected_keys[:10],
    }


class SegmentationService:
    """
    Сервис сегментации изображения.

    Сервис использует модель первого эксперимента и выполняет четыре пользовательских
    операции: получение маски, прозрачный фон, размытие фона и замену фона.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        bundle: SegmentationModelBundle | None = None,
    ) -> None:
        self.config = config or get_config()
        self.bundle = bundle or get_segmentation_bundle()

    def predict_mask(
        self,
        image: Image.Image,
        postprocess: bool = True,
    ) -> tuple[Image.Image, np.ndarray]:
        """
        Строит бинарную маску объекта.

        Возвращает маску в исходном размере изображения и карту вероятностей
        в рабочем размере модели.
        """
        original_size = image.size

        prepared_image, _ = resize_for_model(
            image=image,
            image_size=self.bundle.image_size,
            keep_aspect_ratio=False,
        )

        tensor = tensor_from_image(
            image=prepared_image,
            image_mean=self.bundle.image_mean,
            image_std=self.bundle.image_std,
            device=self.bundle.device,
        )

        self.bundle.model.eval()

        with torch.no_grad():
            output = self.bundle.model(tensor)

        probability_map = probability_map_from_output(output)

        mask_model_size = create_mask_from_probability(
            probability_map=probability_map,
            threshold=self.bundle.threshold,
            postprocess=postprocess,
        )

        mask_original_size = resize_mask_back(mask_model_size, original_size)
        mask_original_size = harden_mask(mask_original_size)

        return mask_original_size, probability_map

    def apply_mode(
        self,
        image: Image.Image,
        mask: Image.Image,
        mode: str,
        background_variant: str = "soft_blue",
        blur_radius: int = 18,
    ) -> Image.Image:
        """
        Применяет выбранный режим обработки к изображению.
        """
        normalized_mode = normalize_segmentation_mode(
            mode,
            default=self.config.segmentation.default_mode,
        )

        if normalized_mode == "binary_mask":
            return create_mask_preview(mask)

        if normalized_mode == "transparent_background":
            return apply_transparent_background(
                image=image,
                mask=mask,
                smooth_edges=True,
            )

        if normalized_mode == "blur_background":
            return apply_blur_background(
                image=image,
                mask=mask,
                blur_radius=blur_radius,
                smooth_edges=True,
            )

        if normalized_mode == "replace_background":
            return apply_generated_background(
                image=image,
                mask=mask,
                background_variant=background_variant,
                smooth_edges=True,
            )

        if normalized_mode == "highlight_object":
            return highlight_object(
                image=image,
                mask=mask,
            )

        raise SegmentationServiceError(f"Неподдерживаемый режим сегментации: {mode}")

    def process_image(
        self,
        input_path: str | Path,
        mode: str | None = None,
        background_variant: str = "soft_blue",
        blur_radius: int = 18,
        postprocess_mask: bool = True,
    ) -> SegmentationResult:
        """
        Выполняет полный сценарий сегментации и сохраняет файлы результата.
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise SegmentationServiceError(f"Исходное изображение не найдено: {input_path}")

        normalized_mode = normalize_segmentation_mode(
            mode,
            default=self.config.segmentation.default_mode,
        )

        source_filename = input_path.name
        original_image = load_image(input_path)

        mask, _probability_map = self.predict_mask(
            image=original_image,
            postprocess=postprocess_mask,
        )

        processed_image = self.apply_mode(
            image=original_image,
            mask=mask,
            mode=normalized_mode,
            background_variant=background_variant,
            blur_radius=blur_radius,
        )

        mask_preview = create_mask_preview(mask)

        mask_info = save_runtime_image(
            image=mask_preview,
            folder=self.config.runtime.result_folder,
            source_filename=source_filename,
            suffix="segmentation_mask",
            extension="png",
            config=self.config,
        )

        result_extension = "png"

        result_info = save_runtime_image(
            image=processed_image,
            folder=self.config.runtime.result_folder,
            source_filename=source_filename,
            suffix=f"segmentation_{normalized_mode}",
            extension=result_extension,
            config=self.config,
        )

        preview_result_image = rgba_to_preview_rgb(processed_image)

        preview = create_side_by_side(
            left_image=original_image,
            right_image=preview_result_image,
            left_title="Исходное изображение",
            right_title=SEGMENTATION_MODE_NAMES_RU.get(
                normalized_mode,
                "Результат сегментации",
            ),
        )

        preview_info = save_runtime_image(
            image=preview,
            folder=self.config.runtime.preview_folder,
            source_filename=source_filename,
            suffix=f"preview_segmentation_{normalized_mode}",
            extension="png",
            config=self.config,
        )

        mask_stats = calculate_mask_stats(mask)
        warnings = self._build_warnings(mask_stats)

        return SegmentationResult(
            mode=normalized_mode,
            mode_name_ru=SEGMENTATION_MODE_NAMES_RU.get(normalized_mode, normalized_mode),
            input_path=str(input_path),
            input_url=path_to_url(input_path, self.config),
            result_image=result_info,
            mask_image=mask_info,
            preview_image=preview_info,
            mask_stats=mask_stats,
            model_info=build_segmentation_model_info(self.bundle),
            warnings=warnings,
        )

    def process_pil_image_without_saving(
        self,
        image: Image.Image,
        mode: str | None = None,
        background_variant: str = "soft_blue",
        blur_radius: int = 18,
        postprocess_mask: bool = True,
    ) -> tuple[Image.Image, Image.Image, SegmentationMaskStats]:
        """
        Выполняет сегментацию без сохранения файлов.

        Метод нужен для полного конвейера, где восстановленное изображение затем
        передается в сегментацию внутри одного пользовательского сценария.
        """
        normalized_mode = normalize_segmentation_mode(
            mode,
            default=self.config.segmentation.default_mode,
        )

        mask, _probability_map = self.predict_mask(
            image=image,
            postprocess=postprocess_mask,
        )

        processed_image = self.apply_mode(
            image=image,
            mask=mask,
            mode=normalized_mode,
            background_variant=background_variant,
            blur_radius=blur_radius,
        )

        return processed_image, mask, calculate_mask_stats(mask)

    def _build_warnings(self, mask_stats: SegmentationMaskStats) -> list[str]:
        """
        Формирует предупреждения о возможных проблемах результата.
        """
        warnings: list[str] = []

        if mask_stats.foreground_pixel_count == 0:
            warnings.append(
                "Модель не выделила объект. Возможно, на изображении нет животного "
                "или объект сильно отличается от обучающих примеров."
            )
            return warnings

        if mask_stats.foreground_percent < 2.0:
            warnings.append(
                "Выделенная область очень мала. Результат сегментации нужно проверить визуально."
            )

        if mask_stats.foreground_percent > 92.0:
            warnings.append(
                "Выделенная область занимает почти все изображение. Возможно, фон отделен неточно."
            )

        return warnings

    def get_model_card(self) -> dict[str, Any]:
        """
        Возвращает карточку модели для страницы о системе.
        """
        model_info = build_segmentation_model_info(self.bundle)

        labels = self.bundle.labels or {}
        classes = labels.get("classes", [])

        return {
            "title": "Модель сегментации объекта",
            "description": (
                self.bundle.metadata.get("description")
                or "Модель выделяет объект животного на изображении."
            ),
            "model_info": model_info,
            "classes": classes,
            "available_modes": get_available_segmentation_modes(),
            "background_variants": get_available_background_variants(),
        }


def get_segmentation_service() -> SegmentationService:
    """
    Создает сервис сегментации с кэшированной моделью.
    """
    return SegmentationService()


def run_segmentation(
    input_path: str | Path,
    mode: str | None = None,
    background_variant: str = "soft_blue",
    blur_radius: int = 18,
) -> SegmentationResult:
    """
    Удобная функция для вызова сегментации из app.py.
    """
    service = get_segmentation_service()

    return service.process_image(
        input_path=input_path,
        mode=mode,
        background_variant=background_variant,
        blur_radius=blur_radius,
    )
