"""Grad-CAM utilities for the repository's exported VGG16 SavedModel."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


class GradCAMError(RuntimeError):
    """Raised when the saved model cannot support a verified Grad-CAM."""


@dataclass(frozen=True)
class GradCAMResult:
    heatmap: np.ndarray
    pneumonia_probability: float
    predicted_class: str
    confidence: float
    target_class: str
    target_index: int | None
    output_index: int
    target_logit: float
    feature_map_shape: tuple[int, ...]
    gradient_shape: tuple[int, ...]
    heatmap_min: float
    heatmap_max: float


HIGH_BIT_DEPTH_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}


def normalize_to_rgb(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and convert to 8-bit RGB.

    Pillow clips 16-bit and 32-bit pixel values to 255 when converting to RGB,
    which turns most high-bit-depth X-rays white, so those are min-max scaled to
    8-bit first.
    """
    image = ImageOps.exif_transpose(image)
    if image.mode in HIGH_BIT_DEPTH_MODES:
        values = np.asarray(image, dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        if high > low:
            values = (values - low) / (high - low) * 255.0
        else:
            values = np.zeros_like(values)
        image = Image.fromarray(np.uint8(np.round(values)), mode="L")
    return image.convert("RGB")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Apply the same RGB, 150x150, [0, 1] preprocessing as app inference."""
    resized = image.convert("RGB").resize((150, 150))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def get_target_layer_name(saved_model: tf.Module) -> str:
    """Find and verify the final VGG16 convolution in the exported graph."""
    signature = saved_model.signatures.get("serving_default")
    if signature is None:
        raise GradCAMError("The SavedModel has no serving_default signature.")

    graph_def = signature.graph.as_graph_def()
    conv_nodes = [
        node
        for function in graph_def.library.function
        for node in function.node_def
        if node.op == "Conv2D"
    ]
    if not conv_nodes:
        raise GradCAMError("No convolutional layers were found in the SavedModel.")

    layer_names = []
    for node in conv_nodes:
        match = re.search(r"/(block\d+_conv\d+)_\d+/convolution$", node.name)
        if match is None:
            raise GradCAMError(
                f"Unexpected convolution node in the exported VGG16 graph: {node.name}"
            )
        layer_names.append(match.group(1))

    expected_layers = [
        f"block{block}_conv{convolution}"
        for block, count in ((1, 2), (2, 2), (3, 3), (4, 3), (5, 3))
        for convolution in range(1, count + 1)
    ]
    if layer_names != expected_layers:
        raise GradCAMError(
            "The exported convolution sequence does not match the VGG16 architecture."
        )

    final_conv = conv_nodes[-1]
    final_name = layer_names[-1]
    activation_name = re.sub(r"/convolution$", "/Relu", final_conv.name)
    function_nodes = {
        node.name: node
        for function in graph_def.library.function
        for node in function.node_def
    }
    activation = function_nodes.get(activation_name)
    if activation is None or activation.op != "Relu":
        raise GradCAMError(
            f"The final convolution {final_name} is not followed by its VGG16 ReLU."
        )
    if not any(
        node.op == "MaxPool" and activation_name + ":activations:0" in node.input
        for node in function_nodes.values()
    ):
        raise GradCAMError(
            f"The final convolution {final_name} is not connected to the classifier."
        )
    return final_name


def _variable_weights(saved_model: tf.Module) -> dict[str, np.ndarray]:
    variables = getattr(saved_model, "variables", None)
    if not variables:
        raise GradCAMError("The SavedModel does not expose its trained variables.")
    return {
        variable.name.rsplit(":", 1)[0]: variable.numpy()
        for variable in variables
    }


def _find_weight(weights: dict[str, np.ndarray], layer_name: str, kind: str) -> np.ndarray:
    suffix = f"/{layer_name}/{kind}"
    exact_name = f"{layer_name}/{kind}"
    matches = [
        value
        for name, value in weights.items()
        if name == exact_name or name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise GradCAMError(
            f"Could not uniquely identify the saved {kind} for {layer_name}."
        )
    return matches[0]


def _build_keras_view(saved_model: tf.Module, target_layer_name: str) -> tf.keras.Model:
    """Rebuild the verified inference graph using the SavedModel's variables."""
    signature = saved_model.signatures["serving_default"]
    input_spec = next(iter(signature.structured_input_signature[1].values()))
    if input_spec.shape.as_list() != [None, 150, 150, 3]:
        raise GradCAMError(
            f"Expected model input shape (None, 150, 150, 3), got {input_spec.shape}."
        )

    weights = _variable_weights(saved_model)
    inputs = tf.keras.Input(shape=(150, 150, 3), name="gradcam_input")
    vgg = tf.keras.applications.VGG16(
        include_top=False,
        weights=None,
        input_tensor=inputs,
    )

    conv_names = [
        layer.name
        for layer in vgg.layers
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    if not conv_names or conv_names[-1] != target_layer_name:
        raise GradCAMError(
            f"Detected target layer {target_layer_name!r} does not match the "
            "rebuilt VGG16 convolution stack."
        )

    for layer_name in conv_names:
        layer = vgg.get_layer(layer_name)
        layer.set_weights(
            [
                _find_weight(weights, layer_name, "kernel"),
                _find_weight(weights, layer_name, "bias"),
            ]
        )

    dense_kernel_names = [
        name
        for name in weights
        if re.search(r"/dense_\d+/kernel$", name)
    ]
    dense_layers = {
        re.search(r"/(dense_\d+)/kernel$", name).group(1)
        for name in dense_kernel_names
    }
    if len(dense_layers) != 2:
        raise GradCAMError(
            "Expected the exported VGG16 classifier to contain two dense layers."
        )

    signature_graph = signature.graph.as_graph_def()
    sigmoid_nodes = [
        node
        for function in signature_graph.library.function
        for node in function.node_def
        if node.op == "Sigmoid"
    ]
    softmax_nodes = [
        node
        for function in signature_graph.library.function
        for node in function.node_def
        if node.op == "Softmax"
    ]
    if len(sigmoid_nodes) != 1 or softmax_nodes:
        raise GradCAMError(
            "Expected exactly one sigmoid output and no softmax output."
        )
    output_match = re.search(
        r"/(dense_\d+)_\d+/Sigmoid$",
        sigmoid_nodes[0].name,
    )
    if output_match is None:
        raise GradCAMError("Could not identify the sigmoid classifier layer.")
    output_layer_name = output_match.group(1)
    hidden_layer_names = dense_layers - {output_layer_name}
    if len(hidden_layer_names) != 1:
        raise GradCAMError("Could not identify the classifier's hidden dense layer.")
    hidden_layer_name = hidden_layer_names.pop()

    first_kernel = _find_weight(weights, hidden_layer_name, "kernel")
    first_bias = _find_weight(weights, hidden_layer_name, "bias")
    output_kernel = _find_weight(weights, output_layer_name, "kernel")
    output_bias = _find_weight(weights, output_layer_name, "bias")
    if output_kernel.shape[-1] != 1 or output_bias.shape != (1,):
        raise GradCAMError("The classifier is not a single-output binary model.")

    features = tf.keras.layers.GlobalAveragePooling2D(
        name="gradcam_global_average_pooling"
    )(vgg.output)
    features = tf.keras.layers.Dense(
        int(first_kernel.shape[-1]),
        activation="relu",
        name="gradcam_hidden",
    )(features)
    logits = tf.keras.layers.Dense(
        1,
        name="gradcam_logits",
    )(features)
    predictions = tf.keras.layers.Activation(
        "sigmoid",
        name="gradcam_prediction",
    )(logits)
    keras_model = tf.keras.Model(
        inputs=inputs,
        outputs=[logits, predictions],
    )
    keras_model.get_layer("gradcam_hidden").set_weights([first_kernel, first_bias])
    keras_model.get_layer("gradcam_logits").set_weights(
        [output_kernel, output_bias]
    )
    return keras_model


class GradCAMExplainer:
    """Build and run Grad-CAM using weights read from an existing SavedModel."""

    def __init__(self, saved_model: tf.Module) -> None:
        self.target_layer_name = get_target_layer_name(saved_model)
        self.input_name = next(
            iter(
                saved_model.signatures["serving_default"]
                .structured_input_signature[1]
            )
        )
        with tf.device("/CPU:0"):
            self.model = _build_keras_view(saved_model, self.target_layer_name)
            signature = saved_model.signatures["serving_default"]
            dummy_input = tf.zeros((1, 150, 150, 3), dtype=tf.float32)
            reference = signature(**{self.input_name: dummy_input})
            reference_output = next(iter(reference.values()))
            _, rebuilt_output = self.model(dummy_input, training=False)
            if not np.allclose(
                reference_output.numpy(),
                rebuilt_output.numpy(),
                rtol=1e-4,
                atol=1e-5,
            ):
                raise GradCAMError(
                    "Rebuilt Grad-CAM model predictions do not match the SavedModel."
                )
            self.grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[
                    self.model.get_layer(self.target_layer_name).output,
                    self.model.get_layer("gradcam_logits").output,
                    self.model.get_layer("gradcam_prediction").output,
                ],
            )

    def generate(self, image_array: np.ndarray) -> GradCAMResult:
        """Compute Grad-CAM for the class predicted by the existing sigmoid head."""
        image_array = np.asarray(image_array, dtype=np.float32)
        if image_array.shape != (1, 150, 150, 3):
            raise GradCAMError(
                "Grad-CAM input must have shape (1, 150, 150, 3)."
            )
        if not np.isfinite(image_array).all():
            raise GradCAMError("Grad-CAM input contains invalid pixel values.")

        with tf.device("/CPU:0"):
            input_tensor = tf.convert_to_tensor(image_array)
            with tf.GradientTape() as tape:
                activations, logits, prediction = self.grad_model(
                    input_tensor,
                    training=False,
                )
                pneumonia_probability = prediction[:, 0]
                predicted_as_pneumonia = pneumonia_probability > 0.5
                predicted_class_score = tf.where(
                    predicted_as_pneumonia,
                    logits[:, 0],
                    -logits[:, 0],
                )

            gradients = tape.gradient(
                predicted_class_score,
                activations,
            )
            if gradients is None:
                raise GradCAMError(
                    f"Gradients are unavailable for {self.target_layer_name}."
                )
            if not bool(tf.reduce_all(tf.math.is_finite(gradients)).numpy()):
                raise GradCAMError("The model produced non-finite Grad-CAM gradients.")

            channel_weights = tf.reduce_mean(gradients, axis=(1, 2))
            heatmap = tf.reduce_sum(
                activations[0] * channel_weights[0][tf.newaxis, tf.newaxis, :],
                axis=-1,
            )
            heatmap = tf.maximum(heatmap, 0)
            if not bool(tf.reduce_all(tf.math.is_finite(heatmap)).numpy()):
                raise GradCAMError("The model produced a non-finite Grad-CAM map.")
            maximum = tf.reduce_max(heatmap)
            if float(maximum.numpy()) <= 0:
                raise GradCAMError("The model produced an empty Grad-CAM heatmap.")
            heatmap_min = float(tf.reduce_min(heatmap).numpy())
            heatmap_max = float(maximum.numpy())
            heatmap = heatmap / maximum
            heatmap_array = heatmap.numpy().astype(np.float32)
            probability = float(pneumonia_probability.numpy()[0])
            predicted_class = (
                "PNEUMONIA" if probability > 0.5 else "NORMAL"
            )
            confidence = (
                probability if predicted_class == "PNEUMONIA" else 1.0 - probability
            )
            target_logit = float(predicted_class_score.numpy()[0])

        return GradCAMResult(
            heatmap=heatmap_array,
            pneumonia_probability=probability,
            predicted_class=predicted_class,
            confidence=confidence,
            target_class=predicted_class,
            target_index=None,
            output_index=0,
            target_logit=target_logit,
            feature_map_shape=tuple(int(value) for value in activations.shape),
            gradient_shape=tuple(int(value) for value in gradients.shape),
            heatmap_min=heatmap_min,
            heatmap_max=heatmap_max,
        )


def generate_raw_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Resize the normalized model CAM as grayscale without recoloring its values."""
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise GradCAMError("Heatmap must be a finite two-dimensional array.")
    values = np.clip(values, 0.0, 1.0)
    return Image.fromarray(np.uint8(values * 255), mode="L").resize(
        size,
        Image.Resampling.BILINEAR,
    )


def generate_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Colorize the already computed CAM only for display."""
    raw_heatmap = generate_raw_heatmap(heatmap, size)
    return ImageOps.colorize(
        raw_heatmap,
        black="#101010",
        mid="#d33b20",
        white="#fff27a",
        blackpoint=0,
        whitepoint=255,
        midpoint=128,
    )


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay the colored heatmap on the original image."""
    if not 0.0 <= alpha <= 1.0:
        raise GradCAMError("Overlay alpha must be between 0 and 1.")
    original = image.convert("RGB")
    colored_heatmap = generate_heatmap(heatmap, original.size)
    return Image.blend(original, colored_heatmap, alpha)


def image_to_data_url(image: Image.Image, format_name: str = "PNG") -> str:
    """Encode an image as a data URL without writing an artifact to disk."""
    buffer = BytesIO()
    image.save(buffer, format=format_name)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime_type = "image/png" if format_name.upper() == "PNG" else "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"
