"""Unit tests for image pre-processing."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from gradcam import normalize_to_rgb, preprocess_image


def test_preprocess_image_gives_model_input_shape_and_range():
    image = Image.new("RGB", (640, 480), (255, 128, 0))

    batch = preprocess_image(image)

    assert batch.shape == (1, 150, 150, 3)
    assert batch.dtype == np.float32
    assert batch.min() >= 0.0 and batch.max() <= 1.0
    np.testing.assert_allclose(batch[0, 0, 0], [1.0, 128 / 255, 0.0], atol=1e-6)


def test_grayscale_image_becomes_three_identical_channels():
    rgb = np.asarray(normalize_to_rgb(Image.new("L", (32, 32), 90)))

    assert rgb.shape == (32, 32, 3)
    assert (rgb == 90).all()


def test_16_bit_image_is_scaled_instead_of_clipped():
    # A 12-bit gradient stored in a 16-bit PNG, as exported by many X-ray systems.
    values = np.linspace(0, 4095, 200 * 200).reshape(200, 200).astype(np.uint16)
    image = Image.fromarray(values)
    assert image.mode == "I;16"

    rgb = np.asarray(normalize_to_rgb(image))

    assert rgb.min() == 0 and rgb.max() == 255
    assert (rgb == 255).mean() < 0.01
    assert np.unique(rgb).size > 200


def test_flat_16_bit_image_does_not_divide_by_zero():
    image = Image.fromarray(np.full((20, 20), 1000, dtype=np.uint16))

    rgb = np.asarray(normalize_to_rgb(image))

    assert (rgb == 0).all()


@pytest.mark.parametrize("orientation, expected_size", [(1, (60, 40)), (6, (40, 60))])
def test_exif_orientation_is_applied(orientation, expected_size):
    image = Image.new("RGB", (60, 40))
    exif = image.getexif()
    exif[0x0112] = orientation
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    buffer.seek(0)

    with Image.open(buffer) as uploaded:
        assert normalize_to_rgb(uploaded).size == expected_size
