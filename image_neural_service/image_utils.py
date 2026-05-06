# image_utils.py
from __future__ import annotations

import base64
import io
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from config import AppConfig, get_config, is_allowed_file


IMAGE_EXTENSIONS_WITH_ALPHA = {"png", "webp"}
DEFAULT_JPEG_QUALITY = 95
DEFAULT_WEBP_QUALITY = 95


@dataclass(frozen=True)
class SavedImageInfo:
    """
    Сведения о сохраненном изображении.

    Поле relative_path удобно использовать в шаблонах Flask.
    Поле absolute_path нужно для внутренней обработки файла.
    """

    absolute_path: Path
    relative_path: str
    filename: str
    extension: str
    width: int
    height: int
    mode: str


@dataclass(frozen=True)
class ImageQualityFeatures:
    """
    Простые признаки изображения для пользовательского отчета.

    Эти признаки не являются полноценной эталонной оценкой качества.
    Они применяются для описания пользовательского изображения, когда эталон недоступен.
    """

    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float
    colorfulness: float


class ImageValidationError(ValueError):
    """
    Ошибка проверки пользовательского изображения.
    """


def make_unique_id(prefix: str = "img") -> str:
    """
    Создает короткий уникальный идентификатор для имени файла.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_part = uuid.uuid4().hex[:10]

    return f"{prefix}_{timestamp}_{random_part}"


def normalize_extension(filename: str, default_extension: str = "png") -> str:
    """
    Возвращает расширение файла без точки.

    Если расширение отсутствует, используется расширение по умолчанию.
    """
    if "." not in filename:
        return default_extension.lower().lstrip(".")

    extension = filename.rsplit(".", 1)[1].lower().strip()
    extension = re.sub(r"[^a-z0-9]", "", extension)

    if not extension:
        return default_extension.lower().lstrip(".")

    return extension


def safe_filename(filename: str, fallback_extension: str = "png") -> str:
    """
    Делает имя файла безопасным для сохранения.

    Кириллица и пробелы заменяются на простые символы, чтобы избежать проблем
    при работе в разных операционных системах.
    """
    extension = normalize_extension(filename, fallback_extension)

    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    stem = stem.strip().lower()
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^a-z0-9_\-]+", "", stem)

    if not stem:
        stem = "image"

    return f"{stem}.{extension}"


def make_output_filename(
    source_filename: str,
    suffix: str,
    extension: str | None = None,
) -> str:
    """
    Создает имя выходного файла на основе имени исходного изображения.
    """
    safe_source = safe_filename(source_filename)

    source_extension = normalize_extension(safe_source, "png")
    output_extension = extension.lower().lstrip(".") if extension else source_extension

    stem = safe_source.rsplit(".", 1)[0]
    unique_part = uuid.uuid4().hex[:8]

    return f"{stem}_{suffix}_{unique_part}.{output_extension}"


def ensure_directory(path: Path) -> None:
    """
    Создает каталог, если он отсутствует.
    """
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: str | Path) -> Image.Image:
    """
    Загружает изображение и приводит его к обычному цветному виду.
    """
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    try:
        with Image.open(image_path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ImageValidationError(f"Не удалось открыть изображение: {image_path}") from exc


def load_image_keep_alpha(path: str | Path) -> Image.Image:
    """
    Загружает изображение с сохранением прозрачности, если она есть.
    """
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)

            if image.mode in {"RGBA", "LA"}:
                return image.convert("RGBA")

            return image.convert("RGB")
    except Exception as exc:
        raise ImageValidationError(f"Не удалось открыть изображение: {image_path}") from exc


def validate_image_file(filename: str, config: AppConfig | None = None) -> None:
    """
    Проверяет, что файл имеет допустимое расширение.
    """
    if config is None:
        config = get_config()

    if not filename:
        raise ImageValidationError("Имя файла не указано.")

    if not is_allowed_file(filename, config):
        allowed = ", ".join(config.runtime.allowed_extensions)
        raise ImageValidationError(
            f"Недопустимый формат файла. Разрешены форматы: {allowed}."
        )


def validate_image_size(path: str | Path, max_megapixels: float = 32.0) -> None:
    """
    Проверяет размер изображения.

    Ограничение защищает приложение от слишком больших файлов.
    """
    image_path = Path(path)

    with Image.open(image_path) as image:
        width, height = image.size

    megapixels = width * height / 1_000_000

    if megapixels > max_megapixels:
        raise ImageValidationError(
            f"Изображение слишком большое: {megapixels:.1f} мегапикселей. "
            f"Максимально допустимо: {max_megapixels:.1f} мегапикселей."
        )


def save_uploaded_file(
    file_storage,
    upload_folder: Path | None = None,
    config: AppConfig | None = None,
) -> SavedImageInfo:
    """
    Сохраняет файл, загруженный через Flask.

    file_storage — объект из request.files.
    """
    if config is None:
        config = get_config()

    if upload_folder is None:
        upload_folder = config.runtime.upload_folder

    ensure_directory(upload_folder)

    original_filename = file_storage.filename or ""
    validate_image_file(original_filename, config)

    extension = normalize_extension(original_filename, "png")
    filename = make_output_filename(
        source_filename=original_filename,
        suffix="upload",
        extension=extension,
    )

    target_path = upload_folder / filename
    file_storage.save(target_path)

    validate_image_size(target_path)

    image = load_image(target_path)

    return SavedImageInfo(
        absolute_path=target_path,
        relative_path=path_to_url(target_path, config),
        filename=filename,
        extension=extension,
        width=image.width,
        height=image.height,
        mode=image.mode,
    )


def save_pil_image(
    image: Image.Image,
    path: str | Path,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> Path:
    """
    Сохраняет изображение с учетом расширения файла.
    """
    output_path = Path(path)
    ensure_directory(output_path.parent)

    extension = normalize_extension(output_path.name, "png")

    if extension in {"jpg", "jpeg"}:
        image_to_save = image.convert("RGB")
        image_to_save.save(output_path, quality=quality, optimize=True)
    elif extension == "webp":
        image.save(output_path, quality=DEFAULT_WEBP_QUALITY, method=6)
    else:
        image.save(output_path)

    return output_path


def save_runtime_image(
    image: Image.Image,
    folder: Path,
    source_filename: str,
    suffix: str,
    extension: str = "png",
    config: AppConfig | None = None,
) -> SavedImageInfo:
    """
    Сохраняет результат обработки в рабочий каталог приложения.
    """
    if config is None:
        config = get_config()

    ensure_directory(folder)

    filename = make_output_filename(
        source_filename=source_filename,
        suffix=suffix,
        extension=extension,
    )

    output_path = folder / filename
    save_pil_image(image, output_path)

    return SavedImageInfo(
        absolute_path=output_path,
        relative_path=path_to_url(output_path, config),
        filename=filename,
        extension=extension,
        width=image.width,
        height=image.height,
        mode=image.mode,
    )


def path_to_url(path: str | Path, config: AppConfig | None = None) -> str:
    """
    Преобразует путь к файлу в относительный адрес для шаблонов.

    Для файлов runtime используется маршрут /runtime/<раздел>/<файл>.
    """
    if config is None:
        config = get_config()

    file_path = Path(path).resolve()

    runtime_roots = [
        ("uploads", config.runtime.upload_folder.resolve()),
        ("results", config.runtime.result_folder.resolve()),
        ("previews", config.runtime.preview_folder.resolve()),
        ("reports", config.runtime.report_folder.resolve()),
    ]

    for route_name, root_path in runtime_roots:
        try:
            relative = file_path.relative_to(root_path)
            return f"/runtime/{route_name}/{relative.as_posix()}"
        except ValueError:
            continue

    try:
        relative_static = file_path.relative_to((config.base_dir / "static").resolve())
        return f"/static/{relative_static.as_posix()}"
    except ValueError:
        return file_path.as_posix()


def image_to_numpy_rgb(image: Image.Image) -> np.ndarray:
    """
    Переводит изображение PIL в массив RGB.
    """
    return np.array(image.convert("RGB"))


def numpy_to_pil_rgb(array: np.ndarray) -> Image.Image:
    """
    Переводит массив в изображение RGB.
    """
    array = np.asarray(array)

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")

    return Image.fromarray(array[:, :, :3], mode="RGB")


def pil_to_base64(image: Image.Image, image_format: str = "PNG") -> str:
    """
    Кодирует изображение в строку base64.

    Может использоваться для небольших предварительных изображений в интерфейсе.
    """
    buffer = io.BytesIO()

    if image_format.upper() in {"JPG", "JPEG"}:
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        mime_type = "image/jpeg"
    else:
        image.save(buffer, format="PNG")
        mime_type = "image/png"

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


def resize_for_model(
    image: Image.Image,
    image_size: int,
    keep_aspect_ratio: bool = False,
) -> tuple[Image.Image, tuple[int, int]]:
    """
    Подготавливает изображение для подачи в модель.

    Если keep_aspect_ratio равно False, изображение просто приводится к квадрату.
    Именно такой режим использовался в экспериментах с рабочим размером 256 на 256.
    """
    original_size = image.size
    image = image.convert("RGB")

    if not keep_aspect_ratio:
        resized = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        return resized, original_size

    fitted = ImageOps.contain(image, (image_size, image_size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))

    left = (image_size - fitted.width) // 2
    top = (image_size - fitted.height) // 2
    canvas.paste(fitted, (left, top))

    return canvas, original_size


def resize_back(image: Image.Image, original_size: tuple[int, int]) -> Image.Image:
    """
    Возвращает изображение к исходному размеру.
    """
    return image.resize(original_size, Image.Resampling.BICUBIC)


def resize_mask_back(mask: Image.Image, original_size: tuple[int, int]) -> Image.Image:
    """
    Возвращает маску к исходному размеру.

    Для маски используется ближайший сосед, чтобы не создавать лишние классы.
    """
    return mask.resize(original_size, Image.Resampling.NEAREST)


def create_binary_mask(
    probability_map: np.ndarray,
    threshold: float = 0.5,
) -> Image.Image:
    """
    Создает бинарную маску из карты вероятностей.

    Значения выше порога относятся к объекту.
    """
    probability_map = np.asarray(probability_map)

    if probability_map.ndim == 3:
        probability_map = probability_map.squeeze()

    binary = (probability_map >= threshold).astype(np.uint8) * 255

    return Image.fromarray(binary, mode="L")


def smooth_mask(mask: Image.Image, radius: float = 1.5) -> Image.Image:
    """
    Слегка сглаживает край маски.
    """
    return mask.convert("L").filter(ImageFilter.GaussianBlur(radius=radius))


def harden_mask(mask: Image.Image, threshold: int = 128) -> Image.Image:
    """
    Возвращает маску к строгим значениям 0 и 255.
    """
    mask_array = np.array(mask.convert("L"))
    binary = (mask_array >= threshold).astype(np.uint8) * 255

    return Image.fromarray(binary, mode="L")


def refine_mask(mask: Image.Image, smooth_radius: float = 1.2) -> Image.Image:
    """
    Простая постобработка маски.

    Маска слегка сглаживается, затем снова переводится в бинарный вид.
    """
    smoothed = smooth_mask(mask, radius=smooth_radius)
    refined = harden_mask(smoothed, threshold=128)

    return refined


def apply_transparent_background(
    image: Image.Image,
    mask: Image.Image,
    smooth_edges: bool = True,
) -> Image.Image:
    """
    Оставляет объект на прозрачном фоне.
    """
    rgb_image = image.convert("RGB")
    mask_l = mask.convert("L").resize(rgb_image.size, Image.Resampling.BICUBIC)

    if smooth_edges:
        mask_l = smooth_mask(mask_l, radius=1.0)

    result = Image.new("RGBA", rgb_image.size, (0, 0, 0, 0))
    result.paste(rgb_image.convert("RGBA"), (0, 0), mask_l)

    return result


def apply_blur_background(
    image: Image.Image,
    mask: Image.Image,
    blur_radius: int = 18,
    smooth_edges: bool = True,
) -> Image.Image:
    """
    Размывает фон и сохраняет объект четким.
    """
    rgb_image = image.convert("RGB")
    mask_l = mask.convert("L").resize(rgb_image.size, Image.Resampling.BICUBIC)

    if smooth_edges:
        mask_l = smooth_mask(mask_l, radius=1.0)

    blurred = rgb_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return Image.composite(rgb_image, blurred, mask_l)


def make_generated_background(
    size: tuple[int, int],
    variant: str = "soft_blue",
) -> Image.Image:
    """
    Создает простой фон без обращения к внешним генеративным сервисам.

    Такой фон нужен для режима замены фона в пользовательском приложении.
    """
    width, height = size

    variants = {
        "soft_blue": ((230, 239, 255), (250, 252, 255)),
        "warm_light": ((255, 241, 225), (255, 252, 247)),
        "gray_studio": ((232, 235, 239), (250, 250, 251)),
        "green_soft": ((226, 244, 232), (248, 253, 249)),
    }

    top_color, bottom_color = variants.get(variant, variants["soft_blue"])

    background = Image.new("RGB", size)
    draw = ImageDraw.Draw(background)

    for y in range(height):
        ratio = y / max(height - 1, 1)

        red = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        green = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        blue = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)

        draw.line([(0, y), (width, y)], fill=(red, green, blue))

    return background


def apply_generated_background(
    image: Image.Image,
    mask: Image.Image,
    background_variant: str = "soft_blue",
    smooth_edges: bool = True,
) -> Image.Image:
    """
    Заменяет фон на простой сгенерированный фон.
    """
    rgb_image = image.convert("RGB")
    mask_l = mask.convert("L").resize(rgb_image.size, Image.Resampling.BICUBIC)

    if smooth_edges:
        mask_l = smooth_mask(mask_l, radius=1.0)

    background = make_generated_background(rgb_image.size, background_variant)

    return Image.composite(rgb_image, background, mask_l)


def highlight_object(
    image: Image.Image,
    mask: Image.Image,
    overlay_color: tuple[int, int, int] = (59, 130, 246),
    alpha: float = 0.35,
) -> Image.Image:
    """
    Подсвечивает найденный объект цветным слоем.
    """
    rgb_image = image.convert("RGB")
    mask_l = mask.convert("L").resize(rgb_image.size, Image.Resampling.NEAREST)

    overlay = Image.new("RGB", rgb_image.size, overlay_color)
    highlighted = Image.blend(rgb_image, overlay, alpha)

    return Image.composite(highlighted, rgb_image, mask_l)


def create_mask_preview(
    mask: Image.Image,
    foreground_color: tuple[int, int, int] = (255, 255, 255),
    background_color: tuple[int, int, int] = (30, 41, 59),
) -> Image.Image:
    """
    Создает цветное изображение маски для просмотра в интерфейсе.
    """
    mask_l = mask.convert("L")
    foreground = Image.new("RGB", mask_l.size, foreground_color)
    background = Image.new("RGB", mask_l.size, background_color)

    return Image.composite(foreground, background, mask_l)


def create_side_by_side(
    left_image: Image.Image,
    right_image: Image.Image,
    left_title: str = "До обработки",
    right_title: str = "После обработки",
    width: int = 1200,
) -> Image.Image:
    """
    Создает изображение для сравнения результата до и после обработки.
    """
    left = left_image.convert("RGB")
    right = right_image.convert("RGB")

    panel_width = width // 2
    image_height = int(panel_width * 0.75)
    title_height = 56
    padding = 18

    left_resized = ImageOps.contain(
        left,
        (panel_width - 2 * padding, image_height - 2 * padding),
        Image.Resampling.BICUBIC,
    )

    right_resized = ImageOps.contain(
        right,
        (panel_width - 2 * padding, image_height - 2 * padding),
        Image.Resampling.BICUBIC,
    )

    canvas_height = image_height + title_height
    canvas = Image.new("RGB", (width, canvas_height), (246, 248, 251))

    draw = ImageDraw.Draw(canvas)

    draw.rounded_rectangle(
        [8, 8, panel_width - 8, canvas_height - 8],
        radius=18,
        fill=(255, 255, 255),
        outline=(218, 226, 234),
        width=2,
    )

    draw.rounded_rectangle(
        [panel_width + 8, 8, width - 8, canvas_height - 8],
        radius=18,
        fill=(255, 255, 255),
        outline=(218, 226, 234),
        width=2,
    )

    draw.text((padding + 6, 18), left_title, fill=(32, 43, 56))
    draw.text((panel_width + padding + 6, 18), right_title, fill=(32, 43, 56))

    left_x = (panel_width - left_resized.width) // 2
    left_y = title_height + (image_height - left_resized.height) // 2

    right_x = panel_width + (panel_width - right_resized.width) // 2
    right_y = title_height + (image_height - right_resized.height) // 2

    canvas.paste(left_resized, (left_x, left_y))
    canvas.paste(right_resized, (right_x, right_y))

    return canvas


def create_thumbnail(
    image: Image.Image,
    max_size: tuple[int, int] = (420, 320),
    background_color: tuple[int, int, int] = (246, 248, 251),
) -> Image.Image:
    """
    Создает предварительное изображение для карточек истории.
    """
    image = image.convert("RGB")
    thumbnail = ImageOps.contain(image, max_size, Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", max_size, background_color)
    x = (max_size[0] - thumbnail.width) // 2
    y = (max_size[1] - thumbnail.height) // 2

    canvas.paste(thumbnail, (x, y))

    return canvas


def estimate_image_quality_features(image: Image.Image) -> ImageQualityFeatures:
    """
    Оценивает простые характеристики изображения без эталона.

    Резкость считается через средний градиент яркости.
    Контраст считается через стандартное отклонение яркости.
    Цветность считается по различию цветовых каналов.
    """
    rgb_image = image.convert("RGB")
    array = np.array(rgb_image).astype(np.float32)

    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]

    brightness_map = 0.299 * red + 0.587 * green + 0.114 * blue

    brightness = float(np.mean(brightness_map))
    contrast = float(np.std(brightness_map))

    grad_y = np.abs(np.diff(brightness_map, axis=0)).mean()
    grad_x = np.abs(np.diff(brightness_map, axis=1)).mean()
    sharpness = float((grad_x + grad_y) / 2.0)

    red_green = red - green
    yellow_blue = 0.5 * (red + green) - blue

    colorfulness = float(
        np.sqrt(np.std(red_green) ** 2 + np.std(yellow_blue) ** 2)
        + 0.3 * np.sqrt(np.mean(red_green) ** 2 + np.mean(yellow_blue) ** 2)
    )

    return ImageQualityFeatures(
        width=rgb_image.width,
        height=rgb_image.height,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        colorfulness=colorfulness,
    )


def describe_quality_features(features: ImageQualityFeatures) -> dict[str, str]:
    """
    Переводит численные признаки изображения в понятные текстовые категории.
    """
    if features.brightness < 75:
        brightness_text = "изображение выглядит темным"
    elif features.brightness > 185:
        brightness_text = "изображение выглядит светлым"
    else:
        brightness_text = "яркость находится в среднем диапазоне"

    if features.contrast < 35:
        contrast_text = "контраст выражен слабо"
    elif features.contrast > 75:
        contrast_text = "контраст выражен сильно"
    else:
        contrast_text = "контраст находится в среднем диапазоне"

    if features.sharpness < 5:
        sharpness_text = "изображение может быть размытым"
    elif features.sharpness > 15:
        sharpness_text = "изображение содержит выраженные детали"
    else:
        sharpness_text = "резкость находится в среднем диапазоне"

    if features.colorfulness < 20:
        colorfulness_text = "цветовая насыщенность невысокая"
    elif features.colorfulness > 65:
        colorfulness_text = "цветовая насыщенность высокая"
    else:
        colorfulness_text = "цветовая насыщенность умеренная"

    return {
        "brightness": brightness_text,
        "contrast": contrast_text,
        "sharpness": sharpness_text,
        "colorfulness": colorfulness_text,
    }


def save_text_report(
    text: str,
    folder: Path,
    source_filename: str,
    suffix: str = "report",
    config: AppConfig | None = None,
) -> SavedImageInfo:
    """
    Сохраняет текстовый отчет в рабочий каталог.

    Возвращается SavedImageInfo для единообразия с изображениями.
    Поля width и height для текстового файла равны нулю.
    """
    if config is None:
        config = get_config()

    ensure_directory(folder)

    filename = make_output_filename(
        source_filename=source_filename,
        suffix=suffix,
        extension="md",
    )

    output_path = folder / filename

    with output_path.open("w", encoding="utf-8") as file:
        file.write(text)

    return SavedImageInfo(
        absolute_path=output_path,
        relative_path=path_to_url(output_path, config),
        filename=filename,
        extension="md",
        width=0,
        height=0,
        mode="text",
    )


def read_text_file(path: str | Path, default: str = "") -> str:
    """
    Читает текстовый файл.
    """
    file_path = Path(path)

    if not file_path.exists():
        return default

    return file_path.read_text(encoding="utf-8")


def read_json_file(path: str | Path, default: dict | None = None) -> dict:
    """
    Читает файл JSON.
    """
    import json

    file_path = Path(path)

    if default is None:
        default = {}

    if not file_path.exists():
        return default

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json_file(data: dict, path: str | Path) -> Path:
    """
    Сохраняет словарь в файл JSON.
    """
    import json

    file_path = Path(path)
    ensure_directory(file_path.parent)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    return file_path
