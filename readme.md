Pneumonia Detection AI

AI-powered web application for detecting Pneumonia from Chest X-ray images using Deep Learning and TensorFlow.

Features
Upload Chest X-ray images
Predict Pneumonia or Normal
Confidence score display
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

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py
Usage
Open browser
Go to:
http://127.0.0.1:5000
Upload a Chest X-ray image
Click Predict
View prediction and confidence score
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
Add Grad-CAM visualization
Multi-disease detection
Author

Aditi Gupta