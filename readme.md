# Pneumonia Detection AI

An AI-powered web application that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a VGG16 transfer-learning model, explains each prediction with a Grad-CAM heat-map, and produces a downloadable PDF report.

> **Research use only.** This is a research and demonstration project, not a medical device. See the [Research Use Notice](#research-use-notice).

## Features

- Upload a chest X-ray (PNG or JPEG, up to 10 MB) and preview it before analysis
- Predict NORMAL or PNEUMONIA with the model's output value (labelled "Prediction confidence" in the app; it is not a calibrated probability)
- Grad-CAM heat-map and overlay showing which image regions influenced the prediction
- Downloadable PDF report with the prediction, score, X-ray, Grad-CAM images and a research-use notice
- Upload checks in the browser **and** on the server: file type, size, pixel dimensions and corrupt files
- Support for 16-bit grayscale PNGs (scaled to 8-bit before prediction) and EXIF-rotated photos
- Nothing is stored: uploads are processed in memory only

## Tech Stack

Python, Flask, TensorFlow / Keras, NumPy, Pillow, ReportLab, HTML, CSS, JavaScript, pytest

## Project Structure

```
Pneumonia-Detection-AI/
├── app.py                    # Flask app: routes, upload validation, PDF report
├── gradcam.py                # Pre-processing and Grad-CAM explainer
├── test_gradcam.py           # Manual Grad-CAM check against the real model
├── saved_pneumonia_model/    # Exported VGG16 TensorFlow SavedModel (weights via Git LFS)
├── templates/
│   └── index.html            # Single-page interface
├── static/
│   └── style.css
├── tests/                    # Automated tests (pytest)
│   ├── conftest.py
│   ├── test_preprocessing.py
│   └── test_routes.py
├── requirements.txt          # Pinned runtime dependencies
├── requirements-dev.txt      # Runtime dependencies + pytest
├── pytest.ini
└── readme.md
```

## Installation

**Requirements:** Python 3.10–3.13 and [Git LFS](https://git-lfs.com) (the model weights are stored with LFS).

```bash
git lfs install
git clone https://github.com/AditiGupta2412/Pneumonia-Detection-AI.git
cd Pneumonia-Detection-AI
```

Create a virtual environment and install the pinned dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the App

```bash
python app.py
```

Then open <http://127.0.0.1:5000> in a browser.

- The model is found relative to `app.py`, so the app can be started from any folder.
- Debug mode is off by default. Set `FLASK_DEBUG=1` to enable it for local development only; the debugger allows code to be run on the server.

## Usage

1. Choose a chest X-ray image (PNG or JPEG).
2. Check the preview and click **Analyze image**.
3. Read the predicted class and the model's score.
4. Click **Generate Explanation** to see the original image, the raw Grad-CAM map and the overlay.
5. Click **Download PDF report** to save a report of the result.

## API

| Route | Method | Input | Success response |
|---|---|---|---|
| `/` | GET | – | HTML page |
| `/predict` | POST | form field `file` | JSON: `prediction`, `confidence`, `score` |
| `/explain` | POST | form field `file` | JSON: prediction, raw Grad-CAM and overlay (PNG data URLs) |
| `/report` | POST | form field `file` | PDF file |

Errors are returned as JSON with an `error` field:

| Status | When |
|---|---|
| 400 | No file, empty file name, format other than PNG/JPEG, corrupt image, or more than 50 million pixels |
| 413 | File larger than 10 MB |
| 500 | Unexpected failure, including an error inside the model |
| 503 | Model not loaded, or Grad-CAM unavailable (`/explain`, `/report`) |

## Testing

### Automated tests

```bash
pip install -r requirements-dev.txt
pytest
```

The 28 tests in `tests/` cover pre-processing (unit), the routes with validation and error handling (integration and API), the 0.5 decision threshold, Grad-CAM responses and PDF generation. TensorFlow is replaced by a stub and the model by a stand-in that returns a chosen score, so the tests run in under a second without loading the model. They check the code, not the quality of the model.

### Grad-CAM check with the real model

```bash
python test_gradcam.py --image "path/to/sample-xray.jpg"
```

This script loads the real SavedModel, checks that the Grad-CAM model gives the same prediction, saves `original_xray.png`, `raw_gradcam.png` and `gradcam_overlay.png` to `gradcam_output/`, prints Grad-CAM diagnostics, and confirms through SHA-256 hashes that the model files were not changed.

## Model Information

- **Architecture:** VGG16 convolutional base pre-trained on ImageNet, followed by global average pooling, a dense layer of 256 ReLU units and one sigmoid output (14,846,273 parameters).
- **Input:** 150 × 150 × 3, pixel values divided by 255.
- **Output:** one sigmoid score; values above 0.5 are classified as PNEUMONIA, otherwise NORMAL.
- **Pre-processing** (shared by prediction and Grad-CAM): EXIF orientation, 16-bit images min-max scaled to 8-bit, RGB conversion, resize to 150 × 150, divide by 255.
- **Grad-CAM** is computed at the final convolution (`block5_conv3`, a 9 × 9 grid) against the class-specific logit: the pneumonia logit, or its negation for NORMAL. The heat-map is a coarse, relative attribution map, not a lesion segmentation, probability map or evidence of diagnostic accuracy.

Test-set results reported in the project report (624 images: 234 NORMAL, 390 PNEUMONIA):

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Baseline CNN | 80.8 % | 0.770 | 0.987 |
| Regularised CNN | 85.3 % | 0.824 | 0.972 |
| **VGG16 transfer learning (deployed)** | **92.5 %** | **0.917** | **0.967** |

The training code is not part of this repository.

## Dataset

[Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), 5,856 paediatric chest X-rays from Kermany et al. (2018), split into training (5,216), validation (16) and test (624) sets.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `ImportError: DLL load failed ... Application Control policy` on Windows | Windows Smart App Control blocks TensorFlow's unsigned DLLs. Run the app under WSL, Linux or macOS, or turn Smart App Control off (it cannot be turned back on without resetting Windows). |
| Every request returns "Model failed to load" | The repository was cloned without Git LFS, so the weights file is only a pointer. Run `git lfs install` and `git lfs pull`. |
| `pip install` cannot find TensorFlow | Use Python 3.10–3.13; TensorFlow 2.21 has no packages for newer versions. |
| "Address already in use" | Another program is using port 5000. Stop it or change the port in `app.py`. |

## Future Improvements

- Calibrate the model score and flag uncertain images
- Fine-tune more layers or try ResNet, DenseNet or EfficientNet
- Higher-resolution explanations (Grad-CAM++, Score-CAM)
- Drag-and-drop upload and other UI improvements
- Cloud deployment (Render / Hugging Face)
- Multi-disease detection

## Research Use Notice

This application is a research and demonstration project, not a medical device. Its predictions must not be used to diagnose, treat or rule out pneumonia. The displayed score is not a calibrated measure of diagnostic certainty. Grad-CAM highlights regions that contributed to a prediction; it does not show a confirmed pneumonia lesion or a clinical diagnosis. Consult a qualified healthcare professional for medical advice.

## Author

Aditi Gupta
