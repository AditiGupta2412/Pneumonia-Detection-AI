"""Shared fixtures for testing the Flask app without loading the real model.

TensorFlow is replaced by a minimal stub before app.py is imported, so the
web layer is tested on its own: the model and the Grad-CAM explainer are
swapped for stand-ins that return chosen values. Model quality is evaluated
separately, and test_gradcam.py checks Grad-CAM against the real SavedModel.
"""

import sys
import types
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def _stub_tensorflow():
    def load(_path):
        raise RuntimeError("TensorFlow stub: the real model is not loaded in tests.")

    stub = types.ModuleType("tensorflow")
    stub.saved_model = types.SimpleNamespace(load=load)
    stub.constant = lambda value: value
    return stub


sys.modules["tensorflow"] = _stub_tensorflow()

import app as app_module  # noqa: E402
from gradcam import GradCAMResult  # noqa: E402


class StandInOutput:
    def __init__(self, score):
        self.score = score

    def numpy(self):
        return np.array([[self.score]], dtype=np.float32)


class StandInModel:
    """Mimics the SavedModel's serving signature with a fixed sigmoid score."""

    def __init__(self, score=0.8):
        self.score = score
        self.last_input = None
        self.signatures = {"serving_default": self._infer}

    def _infer(self, batch):
        self.last_input = batch
        return {"output_0": StandInOutput(self.score)}


class StandInExplainer:
    target_layer_name = "block5_conv3"

    def generate(self, image_array):
        heatmap = np.linspace(0.0, 1.0, 81, dtype=np.float32).reshape(9, 9)
        return GradCAMResult(
            heatmap=heatmap,
            pneumonia_probability=0.8,
            predicted_class="PNEUMONIA",
            confidence=0.8,
            target_class="PNEUMONIA",
            target_index=None,
            output_index=0,
            target_logit=1.386,
            feature_map_shape=(1, 9, 9, 512),
            gradient_shape=(1, 9, 9, 512),
            heatmap_min=0.0,
            heatmap_max=1.0,
        )


@pytest.fixture
def model(monkeypatch):
    stand_in = StandInModel()
    monkeypatch.setattr(app_module, "model", stand_in)
    monkeypatch.setattr(app_module, "gradcam_explainer", StandInExplainer())
    return stand_in


@pytest.fixture
def client(model):
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def encode_image(image, image_format):
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def upload(client, route, data, filename="xray.png"):
    return client.post(
        route,
        data={"file": (BytesIO(data), filename)},
        content_type="multipart/form-data",
    )
