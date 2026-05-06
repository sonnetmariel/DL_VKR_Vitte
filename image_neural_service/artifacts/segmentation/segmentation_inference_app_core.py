
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
import segmentation_models_pytorch as smp


def create_model(encoder_name):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model


def load_checkpoint_safely(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def load_segmentation_model(model_path, encoder_name="resnet34", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = load_checkpoint_safely(model_path, device)

    model = create_model(encoder_name=checkpoint.get("encoder_name", encoder_name))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint, device


def prepare_image_for_model(pil_image, image_size, image_mean, image_std):
    image = ImageOps.exif_transpose(pil_image.convert("RGB"))
    resized_image = image.resize((image_size, image_size), resample=Image.BILINEAR)

    image_array = np.array(resized_image).astype(np.float32) / 255.0

    mean_array = np.array(image_mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.array(image_std, dtype=np.float32).reshape(1, 1, 3)

    normalized_array = (image_array - mean_array) / std_array
    tensor = torch.from_numpy(normalized_array).permute(2, 0, 1).unsqueeze(0).float()

    return tensor


def predict_mask(model, pil_image, config, device):
    image_size = int(config["image_size"])
    image_mean = config["image_mean"]
    image_std = config["image_std"]
    threshold = float(config["threshold"])

    input_tensor = prepare_image_for_model(
        pil_image=pil_image,
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
    ).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    binary_mask = (probability > threshold).astype(np.uint8)

    return probability, binary_mask


def create_generated_background(width, height):
    x_values = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_values = np.linspace(0.0, 1.0, height, dtype=np.float32)

    x_grid, y_grid = np.meshgrid(x_values, y_values)

    red_channel = 210 + 35 * x_grid
    green_channel = 225 + 25 * y_grid
    blue_channel = 245 - 20 * x_grid + 10 * y_grid

    background = np.stack([red_channel, green_channel, blue_channel], axis=2)
    background = np.clip(background, 0, 255).astype(np.uint8)

    return Image.fromarray(background, mode="RGB")


def process_image_with_mask(pil_image, probability_mask):
    image = ImageOps.exif_transpose(pil_image.convert("RGB"))
    width, height = image.size

    probability_image = Image.fromarray(
        (probability_mask * 255).astype(np.uint8),
        mode="L",
    )
    probability_image = probability_image.resize((width, height), resample=Image.BILINEAR)
    soft_alpha = probability_image.filter(ImageFilter.GaussianBlur(radius=1.0))

    transparent_object = image.convert("RGBA")
    transparent_object.putalpha(soft_alpha)

    blurred_background = image.filter(ImageFilter.GaussianBlur(radius=10))
    image_with_blurred_background = Image.composite(
        image,
        blurred_background,
        soft_alpha,
    )

    generated_background = create_generated_background(width, height)
    image_with_generated_background = Image.composite(
        image,
        generated_background,
        soft_alpha,
    )

    return {
        "transparent_object": transparent_object,
        "blurred_background": image_with_blurred_background,
        "generated_background": image_with_generated_background,
        "alpha_mask": soft_alpha,
    }


def run_full_processing(image_path, model_path, config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint, device = load_segmentation_model(
        model_path=model_path,
        encoder_name=config.get("encoder_name", "resnet34"),
    )

    image = Image.open(image_path).convert("RGB")

    probability, binary_mask = predict_mask(
        model=model,
        pil_image=image,
        config=config,
        device=device,
    )

    processed = process_image_with_mask(
        pil_image=image,
        probability_mask=probability,
    )

    Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L").save(
        output_dir / "predicted_mask.png"
    )
    processed["transparent_object"].save(output_dir / "object_transparent.png")
    processed["blurred_background"].save(output_dir / "background_blurred.png")
    processed["generated_background"].save(output_dir / "background_generated.png")

    return {
        "predicted_mask": output_dir / "predicted_mask.png",
        "object_transparent": output_dir / "object_transparent.png",
        "background_blurred": output_dir / "background_blurred.png",
        "background_generated": output_dir / "background_generated.png",
    }
