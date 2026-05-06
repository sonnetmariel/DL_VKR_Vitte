from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

source_path = Path("control_images/control_abyssinian_original.jpg")
output_dir = Path("control_images")
output_dir.mkdir(exist_ok=True)

image = Image.open(source_path).convert("RGB")
image.thumbnail((1200, 1200))

# Вариант 1. Шум, на нем модель обычно показывает наиболее заметное улучшение.
array = np.array(image).astype(np.float32)
random_generator = np.random.default_rng(42)
noise = random_generator.normal(loc=0, scale=22, size=array.shape)
noisy_array = np.clip(array + noise, 0, 255).astype(np.uint8)
Image.fromarray(noisy_array).save(output_dir / "control_abyssinian_noise.jpg", quality=94)

# Вариант 2. Размытие, подходит для демонстрации восстановления деталей.
blurred_image = image.filter(ImageFilter.GaussianBlur(radius=2.2))
blurred_image.save(output_dir / "control_abyssinian_blur.jpg", quality=94)

# Вариант 3. Уменьшение и обратное увеличение размера.
small_size = (max(1, image.width // 2), max(1, image.height // 2))
downscaled_image = image.resize(small_size, Image.Resampling.BICUBIC)
upscaled_image = downscaled_image.resize(image.size, Image.Resampling.BICUBIC)
upscaled_image.save(output_dir / "control_abyssinian_downscale.jpg", quality=94)

print("Файлы подготовлены:")
print(output_dir / "control_abyssinian_original.jpg")
print(output_dir / "control_abyssinian_noise.jpg")
print(output_dir / "control_abyssinian_blur.jpg")
print(output_dir / "control_abyssinian_downscale.jpg")
