"""Integration and API tests for the Flask routes, using a stand-in model."""

import numpy as np
import pytest
from PIL import Image

import app as app_module
from conftest import encode_image, upload

XRAY = Image.new("L", (300, 300), 120)


# --- Decision rule ---------------------------------------------------------

@pytest.mark.parametrize(
    "score, expected_class, expected_value",
    [
        (0.8, "PNEUMONIA", 80.0),
        (0.2, "NORMAL", 80.0),
        (0.5, "NORMAL", 50.0),
        (0.51, "PNEUMONIA", 51.0),
    ],
)
def test_predict_applies_threshold(client, model, score, expected_class, expected_value):
    model.score = score

    response = upload(client, "/predict", encode_image(XRAY, "PNG"))

    assert response.status_code == 200
    body = response.get_json()
    assert body["prediction"] == expected_class
    assert body["score"] == pytest.approx(expected_value)


def test_predict_sends_preprocessed_tensor_to_model(client, model):
    upload(client, "/predict", encode_image(XRAY, "PNG"))

    assert model.last_input.shape == (1, 150, 150, 3)
    assert model.last_input.max() <= 1.0


# --- Valid uploads ---------------------------------------------------------

def test_home_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Pneumonia Detection" in response.data


@pytest.mark.parametrize("image_format, filename", [("PNG", "x.png"), ("JPEG", "x.jpg")])
def test_predict_accepts_png_and_jpeg(client, image_format, filename):
    response = upload(client, "/predict", encode_image(XRAY, image_format), filename)

    assert response.status_code == 200


def test_predict_handles_16_bit_png(client, model):
    values = np.linspace(0, 4095, 300 * 300).reshape(300, 300).astype(np.uint16)

    response = upload(client, "/predict", encode_image(Image.fromarray(values), "PNG"))

    assert response.status_code == 200
    assert 0.4 < float(model.last_input.mean()) < 0.6


# --- Invalid uploads -------------------------------------------------------

def test_predict_without_file_returns_400(client):
    response = client.post("/predict", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["error"] == "No file uploaded."


def test_predict_with_empty_filename_returns_400(client):
    response = upload(client, "/predict", b"", filename="")

    assert response.status_code == 400


def test_predict_rejects_other_image_formats(client):
    response = upload(client, "/predict", encode_image(XRAY, "GIF"), "x.gif")

    assert response.status_code == 400
    assert "Unsupported image format" in response.get_json()["error"]


@pytest.mark.parametrize(
    "data",
    [b"this is not an image", encode_image(XRAY, "PNG")[:100]],
    ids=["random-bytes", "truncated-png"],
)
def test_predict_rejects_corrupt_files(client, data):
    response = upload(client, "/predict", data)

    assert response.status_code == 400
    assert "not a valid image" in response.get_json()["error"]


def test_predict_rejects_images_with_too_many_pixels(client):
    huge = encode_image(Image.new("1", (8000, 8000)), "PNG")

    response = upload(client, "/predict", huge)

    assert response.status_code == 400
    assert "too large" in response.get_json()["error"]


def test_predict_rejects_files_over_10_mb(client):
    response = upload(client, "/predict", b"0" * (10 * 1024 * 1024 + 1))

    assert response.status_code == 413


# --- Failures --------------------------------------------------------------

def test_predict_returns_503_when_model_is_missing(client, monkeypatch):
    monkeypatch.setattr(app_module, "model", None)

    response = upload(client, "/predict", encode_image(XRAY, "PNG"))

    assert response.status_code == 503


def test_model_error_returns_500_not_invalid_image(client, model):
    def broken_model(_batch):
        raise ValueError("shape mismatch inside the model")

    model.signatures["serving_default"] = broken_model

    response = upload(client, "/predict", encode_image(XRAY, "PNG"))

    assert response.status_code == 500
    assert "not a valid image" not in response.get_json()["error"]


# --- Explanation and report ------------------------------------------------

def test_explain_returns_heatmap_and_overlay(client):
    response = upload(client, "/explain", encode_image(XRAY, "PNG"))

    assert response.status_code == 200
    body = response.get_json()
    assert body["prediction"] == "PNEUMONIA"
    assert body["target_layer"] == "block5_conv3"
    assert body["raw_gradcam"].startswith("data:image/png;base64,")
    assert body["overlay"].startswith("data:image/png;base64,")


def test_explain_returns_503_when_explainer_is_unavailable(client, monkeypatch):
    monkeypatch.setattr(app_module, "gradcam_explainer", None)

    response = upload(client, "/explain", encode_image(XRAY, "PNG"))

    assert response.status_code == 503


def test_report_returns_pdf(client):
    response = upload(client, "/report", encode_image(XRAY, "JPEG"), "x.jpg")

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF")


def test_report_rejects_invalid_upload(client):
    response = upload(client, "/report", b"not an image")

    assert response.status_code == 400
