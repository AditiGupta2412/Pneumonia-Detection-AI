"""Image pre-processing and Grad-CAM utilities for the exported VGG16 SavedModel.

A TensorFlow SavedModel only exposes its serving signature, which gives the
final score but not the intermediate activations Grad-CAM needs. This module
therefore:

1. inspects the exported graph to confirm it is the expected VGG16 classifier,
2. rebuilds an equivalent Keras model from the SavedModel's own weights,
3. checks that the rebuilt model gives the same output as the SavedModel, and
4. computes Grad-CAM on the rebuilt model.

The pre-processing functions here are also used by app.py for prediction, so
prediction and Grad-CAM always see exactly the same input.
"""

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
    """Output of GradCAMExplainer.generate() for one image."""

    heatmap: np.ndarray                 # 9 x 9 map scaled to [0, 1]
    pneumonia_probability: float        # raw sigmoid output of the model
    predicted_class: str                # "PNEUMONIA" if the output > 0.5, else "NORMAL"
    confidence: float                   # score of the predicted class (0-1), not calibrated
    target_class: str                   # class the heat-map explains (the predicted class)
    target_index: int | None            # always None: a single sigmoid has no class index
    output_index: int                   # index of the single model output (0)
    target_logit: float                 # class-specific logit used for the gradients
    feature_map_shape: tuple[int, ...]  # shape of the last convolution's activations
    gradient_shape: tuple[int, ...]     # shape of the gradients (same as above)
    heatmap_min: float                  # raw map minimum before scaling
    heatmap_max: float                  # raw map maximum before scaling


# Pillow modes for 16-bit and 32-bit images, which need scaling to 8-bit.
HIGH_BIT_DEPTH_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}


def normalize_to_rgb(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and convert to 8-bit RGB.

    Pillow clips 16-bit and 32-bit pixel values to 255 when converting to RGB,
    which turns most high-bit-depth X-rays white, so those are min-max scaled to
    8-bit first.
    """
    # Phone photos store their rotation in EXIF instead of rotating the pixels.
    image = ImageOps.exif_transpose(image)
    if image.mode in HIGH_BIT_DEPTH_MODES:
        # Stretch the image's own range to 0-255. Many X-ray systems use only
        # 12 of the 16 bits, so dividing by 65535 would give a very dark image.
        values = np.asarray(image, dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        if high > low:
            values = (values - low) / (high - low) * 255.0
        else:
            # A uniform image has no range to stretch.
            values = np.zeros_like(values)
        image = Image.fromarray(np.uint8(np.round(values)), mode="L")
    return image.convert("RGB")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Turn an RGB image into the model's input: a (1, 150, 150, 3) float32 batch in [0, 1].

    VGG16 expects three channels, so grayscale X-rays are repeated across RGB.
    150 x 150 is the input size the model was exported with.
    """
    resized = image.convert("RGB").resize((150, 150))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def get_target_layer_name(saved_model: tf.Module) -> str:
    """Find and verify the final VGG16 convolution in the exported graph."""
    signature = saved_model.signatures.get("serving_default")
    if signature is None:
        raise GradCAMError("The SavedModel has no serving_default signature.")

    # The model's layers live inside the graph's function library; collect
    # every convolution in the order it appears.
    graph_def = signature.graph.as_graph_def()
    conv_nodes = [
        node
        for function in graph_def.library.function
        for node in function.node_def
        if node.op == "Conv2D"
    ]
    if not conv_nodes:
        raise GradCAMError("No convolutional layers were found in the SavedModel.")

    # Node names look like ".../block5_conv3_1/convolution"; keep the Keras
    # layer name ("block5_conv3").
    layer_names = []
    for node in conv_nodes:
        match = re.search(r"/(block\d+_conv\d+)_\d+/convolution$", node.name)
        if match is None:
            raise GradCAMError(
                f"Unexpected convolution node in the exported VGG16 graph: {node.name}"
            )
        layer_names.append(match.group(1))

    # VGG16 has 13 convolutions in five blocks of 2, 2, 3, 3 and 3 layers.
    expected_layers = [
        f"block{block}_conv{convolution}"
        for block, count in ((1, 2), (2, 2), (3, 3), (4, 3), (5, 3))
        for convolution in range(1, count + 1)
    ]
    if layer_names != expected_layers:
        raise GradCAMError(
            "The exported convolution sequence does not match the VGG16 architecture."
        )

    # Grad-CAM uses the last convolution. Confirm that its ReLU output really
    # feeds the final max-pooling layer, i.e. that it is connected to the
    # classifier and is not a leftover node.
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
    """Return the SavedModel's trained weights keyed by variable name (without ':0')."""
    variables = getattr(saved_model, "variables", None)
    if not variables:
        raise GradCAMError("The SavedModel does not expose its trained variables.")
    return {
        variable.name.rsplit(":", 1)[0]: variable.numpy()
        for variable in variables
    }


def _find_weight(weights: dict[str, np.ndarray], layer_name: str, kind: str) -> np.ndarray:
    """Return the "kernel" or "bias" array of one layer, failing unless exactly one matches.

    Variable names may carry a model prefix (e.g. "vgg16/block1_conv1/kernel"),
    so both the exact name and a matching suffix are accepted.
    """
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
    """Rebuild the verified inference graph using the SavedModel's variables.

    The rebuilt Keras model has the same layers and weights as the SavedModel:
    VGG16 convolutions -> global average pooling -> dense ReLU -> one logit ->
    sigmoid. Unlike the SavedModel, it exposes the logit and the activations
    of each layer, which Grad-CAM needs.
    """
    signature = saved_model.signatures["serving_default"]
    input_spec = next(iter(signature.structured_input_signature[1].values()))
    if input_spec.shape.as_list() != [None, 150, 150, 3]:
        raise GradCAMError(
            f"Expected model input shape (None, 150, 150, 3), got {input_spec.shape}."
        )

    # Build an empty VGG16 base (no downloaded weights) and copy the trained
    # convolution weights into it from the SavedModel.
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

    # The classifier head has two dense layers: a hidden ReLU layer and the
    # single-unit output layer.
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

    # The output layer is the one followed by the graph's only Sigmoid node;
    # the other dense layer is the hidden layer.
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

    # Recreate the head. The logit and the sigmoid are separate layers so
    # Grad-CAM can take gradients of the logit (before the sigmoid squashes it).
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
    """Build and run Grad-CAM using weights read from an existing SavedModel.

    Construction raises GradCAMError if the rebuilt model does not reproduce
    the SavedModel's output, so a heat-map is never produced for a model other
    than the one used for prediction.
    """

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
            # Safety check: run a blank image through both models and require
            # the same output.
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
            # One forward pass returns everything Grad-CAM needs: the last
            # convolution's activations, the logit and the sigmoid score.
            self.grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[
                    self.model.get_layer(self.target_layer_name).output,
                    self.model.get_layer("gradcam_logits").output,
                    self.model.get_layer("gradcam_prediction").output,
                ],
            )

    def generate(self, image_array: np.ndarray) -> GradCAMResult:
        """Compute Grad-CAM for the class predicted by the existing sigmoid head.

        image_array is the (1, 150, 150, 3) batch from preprocess_image().
        Runs on the CPU, which is fast enough for one image.
        """
        image_array = np.asarray(image_array, dtype=np.float32)
        if image_array.shape != (1, 150, 150, 3):
            raise GradCAMError(
                "Grad-CAM input must have shape (1, 150, 150, 3)."
            )
        if not np.isfinite(image_array).all():
            raise GradCAMError("Grad-CAM input contains invalid pixel values.")

        with tf.device("/CPU:0"):
            input_tensor = tf.convert_to_tensor(image_array)
            # The tape records the forward pass so gradients can be taken
            # with respect to the convolution activations afterwards.
            with tf.GradientTape() as tape:
                activations, logits, prediction = self.grad_model(
                    input_tensor,
                    training=False,
                )
                # A single sigmoid has no separate NORMAL output. The logit
                # measures evidence for PNEUMONIA, so its negative measures
                # evidence for NORMAL; explain whichever class was predicted.
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

            # Grad-CAM: weight each of the 512 feature maps by its average
            # gradient, add them up, and keep only positive evidence (ReLU).
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
            # Scale to [0, 1] for display; the raw range is kept for diagnostics.
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
    """Resize the normalized model CAM as grayscale without recoloring its values.

    White means a stronger relative contribution. Bilinear resizing smooths
    the 9 x 9 grid up to the image size.
    """
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise GradCAMError("Heatmap must be a finite two-dimensional array.")
    values = np.clip(values, 0.0, 1.0)
    return Image.fromarray(np.uint8(values * 255), mode="L").resize(
        size,
        Image.Resampling.BILINEAR,
    )


def generate_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Colorize the already computed CAM only for display.

    Low values become near-black, middle values red and high values yellow.
    """
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
    """Overlay the colored heatmap on the original image.

    alpha is the heat-map's share of the blend: 0.4 keeps the X-ray visible
    underneath.
    """
    if not 0.0 <= alpha <= 1.0:
        raise GradCAMError("Overlay alpha must be between 0 and 1.")
    original = image.convert("RGB")
    colored_heatmap = generate_heatmap(heatmap, original.size)
    return Image.blend(original, colored_heatmap, alpha)


def image_to_data_url(image: Image.Image, format_name: str = "PNG") -> str:
    """Encode an image as a data URL without writing an artifact to disk.

    The web page can use the result directly as an <img> src.
    """
    buffer = BytesIO()
    image.save(buffer, format=format_name)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime_type = "image/png" if format_name.upper() == "PNG" else "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"
