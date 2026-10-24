Pneumonia Detection AI

AI-powered web application for detecting Pneumonia from Chest X-ray images using Deep Learning and TensorFlow.

Features
Upload Chest X-ray images
Predict Pneumonia or Normal
Model score display (not calibrated confidence)
PNG and JPEG upload validation (checked in the browser and on the server) with a 10 MB limit
16-bit grayscale PNG support (scaled to 8-bit before prediction)
Chest X-ray preview before prediction
Optional Grad-CAM heatmap and overlay using the existing trained model
Flask web application
Deep Learning model integration
Real-time image prediction
Tech Stack
Python
Flask
TensorFlow / Keras
NumPy
PIL (Pillow)
HTML
CSS
JavaScript
Project Structure
Pneumonia-Detection/
│
├── app.py
├── gradcam.py
├── test_gradcam.py
├── requirements.txt
├── README.md
├── saved_pneumonia_model/
├── static/
│   └── style.css
├── templates/
│   └── index.html
Installation

Clone the repository:

git clone https://github.com/your-username/pneumonia-detection-ai.git

Go to project folder:

cd pneumonia-detection-ai

Create a virtual environment (Python 3.10–3.13) and install the pinned dependencies:

python -m venv .venv
.venv\Scripts\activate        (Windows)
source .venv/bin/activate     (macOS / Linux)
pip install -r requirements.txt

Run the application:

python app.py

The app finds the model relative to app.py, so it can be started from any folder.
Debug mode is off by default; set FLASK_DEBUG=1 to enable it for local development only.
On Windows with Smart App Control on, TensorFlow's DLLs are blocked and the app cannot start.
Usage
Open browser
Go to:
http://127.0.0.1:5000
Upload a Chest X-ray image
Review the image preview and click Analyze image
View the predicted class and model score
Click Generate Explanation to view the original image, Grad-CAM heatmap, and overlay
Click Download PDF report to save a report with the prediction, model score, X-ray, Grad-CAM images, and research-use notice

Grad-CAM Smoke Test
Run the standalone test with a sample X-ray:

python test_gradcam.py --image "path/to/sample-xray.jpg"

The test verifies the existing model's prediction, detects its final convolution,
generates `original_xray.png`, `raw_gradcam.png`, and `gradcam_overlay.png`
inside `gradcam_output/`, prints Grad-CAM diagnostics, and checks that the
SavedModel files are unchanged. The repository does not include training code
or training preprocessing/class-index metadata; the current inference preprocessing
(EXIF orientation, 16-bit images min-max scaled to 8-bit, RGB, resize to 150x150,
divide by 255) is shared by prediction and Grad-CAM.
The exported model accepts one sigmoid probability; the existing app interprets
values above 0.5 as PNEUMONIA. Grad-CAM is computed against the class-specific
logit (the pneumonia logit, or its negation for NORMAL) at the final convolution.
The single sigmoid output has no separate softmax class index. The heatmap is a
coarse, relative attribution map; it is not a lesion segmentation, probability
map, or evidence of diagnostic accuracy.

Research Use Notice
This application is a research/demo project, not a medical device. Its predictions
must not be used to diagnose, treat, or rule out pneumonia. The displayed model
score is not calibrated diagnostic certainty. Grad-CAM highlights regions that
contributed to a prediction; it does not represent a confirmed pneumonia lesion
or clinical diagnosis.
Consult a qualified healthcare professional for medical advice.
Model Information
Deep Learning based Pneumonia Detection model
Trained on Chest X-ray dataset
Binary Classification:
NORMAL
PNEUMONIA
Dataset

Chest X-ray Images (Pneumonia) Dataset

Future Improvements
Better UI/UX
Drag and drop upload
Model optimization
Deploy on Render / Hugging Face / Streamlit
Evaluate calibration and robustness on a labeled test set
Multi-disease detection
Author

Aditi Gupta