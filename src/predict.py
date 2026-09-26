"""Predict one local dermoscopic image and save a Grad-CAM explanation."""
import argparse
from pathlib import Path

from train import ROOT, FIGURES, IMAGE_SIZE, setup, load_model, load_class_names, melanoma_probabilities
from gradcam import make_heatmap, overlay_heatmap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


def predict_image(image_path, model, names):
    """Shared CLI/UI inference; returns the score and saves report figures."""
    path = Path(image_path).expanduser()
    if not path.is_absolute() and not path.is_file():
        path = ROOT / path
    try:
        with Image.open(path) as image:
            original = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read input image '{path}': {error}") from error
    (ROOT / "outputs/predictions").mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    # Same bilinear resizing and 0..255 pixel range as the training loader.
    batch = tf.image.resize(original, IMAGE_SIZE)[None, ...]
    probability = float(melanoma_probabilities(model(batch, training=False).numpy(), names)[0])
    prediction = names[names.index("melanoma") if probability >= 0.50 else names.index("non_melanoma")]
    heatmap, layer_name = make_heatmap(model, batch, names)
    resized = np.clip(batch.numpy()[0], 0, 255).astype(np.uint8)
    overlay = overlay_heatmap(resized, heatmap)
    output = ROOT / "outputs/predictions/example_gradcam.png"
    Image.fromarray(overlay).save(output)
    np.save(ROOT / "outputs/predictions/example_heatmap.npy", heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(resized)
    axes[0].set_title("Original image (resized)")
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM: melanoma score")
    axes[2].text(0.04, 0.85, f"Prediction: {prediction}\n\nMelanoma probability: {probability:.4f}\nThreshold: 0.50\n\nEducational prototype.\nNot a clinical diagnosis.\nHeatmap is not a lesion boundary.",
                 va="top", fontsize=11, transform=axes[2].transAxes)
    axes[2].set_title("Model output")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "gradcam_example.png", dpi=160)
    plt.close(fig)
    return {"prediction": prediction, "probability": probability, "layer": layer_name,
            "original": resized, "overlay": overlay, "output": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Local JPEG or PNG image path")
    args = parser.parse_args()
    setup()
    try:
        result = predict_image(args.image, load_model(), load_class_names())
    except ValueError as error:
        raise SystemExit(str(error)) from error
    prediction, probability = result["prediction"], result["probability"]
    output, layer_name = result["output"], result["layer"]
    print(f"Prediction: {prediction}")
    print(f"Melanoma probability: {probability:.4f}")
    print(f"Saved Grad-CAM image: {output.relative_to(ROOT).as_posix()}")
    print(f"Explanation layer: {layer_name}")
    print("Educational prototype only; not medical advice or a diagnostic system.")


if __name__ == "__main__":
    main()
