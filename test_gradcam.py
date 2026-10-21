"""Run genuine Grad-CAM on an X-ray and save the raw CAM and overlay."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from gradcam import (
    GradCAMExplainer,
    generate_raw_heatmap,
    overlay_heatmap,
    preprocess_image,
)


def _model_hashes(model_dir: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(path for path in model_dir.rglob("*") if path.is_file()):
        hasher = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                hasher.update(chunk)
        hashes[str(path.relative_to(model_dir))] = hasher.hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("saved_pneumonia_model"),
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("gradcam_output"))
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    if not model_dir.is_dir():
        parser.error(f"Model directory does not exist: {model_dir}")
    if not image_path.is_file():
        parser.error(f"X-ray image does not exist: {image_path}")

    before_hashes = _model_hashes(model_dir)
    saved_model = tf.saved_model.load(str(model_dir))
    signature = saved_model.signatures.get("serving_default")
    if signature is None:
        raise RuntimeError("SavedModel serving_default signature did not load.")

    with Image.open(image_path) as source:
        original = source.convert("RGB")
    batch = preprocess_image(original)

    input_name = next(iter(signature.structured_input_signature[1]))
    reference_result = signature(**{input_name: tf.constant(batch)})
    reference_output = next(iter(reference_result.values()))
    if reference_output.shape != (1, 1):
        raise RuntimeError(
            f"Expected a single sigmoid output, got {reference_output.shape}."
        )
    raw_output = float(reference_output.numpy()[0, 0])
    if not 0.0 <= raw_output <= 1.0:
        raise RuntimeError(f"Sigmoid output is outside [0, 1]: {raw_output}.")
    reference_class = "PNEUMONIA" if raw_output > 0.5 else "NORMAL"

    explainer = GradCAMExplainer(saved_model)
    result = explainer.generate(batch)
    if not np.isclose(raw_output, result.pneumonia_probability, rtol=1e-4, atol=1e-5):
        raise AssertionError("Grad-CAM output differs from the exported model.")
    if result.predicted_class != reference_class:
        raise AssertionError("Grad-CAM target class differs from model prediction.")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_heatmap = generate_raw_heatmap(result.heatmap, original.size)
    overlay = overlay_heatmap(original, result.heatmap)
    original.save(output_dir / "original_xray.png", format="PNG")
    raw_heatmap.save(output_dir / "raw_gradcam.png", format="PNG")
    overlay.save(output_dir / "gradcam_overlay.png", format="PNG")

    after_hashes = _model_hashes(model_dir)
    if before_hashes != after_hashes:
        raise AssertionError("SavedModel files changed during the test.")

    print(f"Predicted class: {result.predicted_class}")
    print(f"Raw model output (sigmoid Pneumonia score): {result.pneumonia_probability:.8f}")
    print(f"Confidence: {result.confidence * 100:.4f}%")
    print(
        f"Selected target class/index: {result.target_class} / "
        f"single sigmoid output {result.output_index} "
        f"(scalar sigmoid has no separate class index; "
        f"class-specific logit {result.target_logit:.8f})"
    )
    print(f"Selected convolutional layer: {explainer.target_layer_name}")
    print(f"Feature-map shape: {result.feature_map_shape}")
    print(f"Gradient shape: {result.gradient_shape}")
    print(
        f"Pre-normalization raw CAM min/max: "
        f"{result.heatmap_min:.8f} / {result.heatmap_max:.8f}"
    )
    print(f"Saved visualizations to: {output_dir}")
    print("SavedModel files are unchanged.")


if __name__ == "__main__":
    main()
