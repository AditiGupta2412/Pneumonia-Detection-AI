from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf

# =========================
# FLASK APP
# =========================

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================

MODEL_PATH = "saved_pneumonia_model"

try:
    print("⏳ Loading Pneumonia Detection Model...")

    model = load_model(MODEL_PATH)

    print("✅ Model loaded successfully!")
    print("🚀 AI System Ready!\n")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# =========================
# HOME ROUTE
# =========================

@app.route('/')
def home():
    return render_template('index.html')

# =========================
# PREDICTION ROUTE
# =========================

@app.route('/predict', methods=['POST'])
def predict():

    # Check model loaded
    if model is None:
        return jsonify({
            'error': 'Model failed to load.'
        })

    # Check file uploaded
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file uploaded.'
        })

    file = request.files['file']

    # Check file selected
    if file.filename == '':
        return jsonify({
            'error': 'No file selected.'
        })

    try:

        # =========================
        # IMAGE PREPROCESSING
        # =========================

        # Open image
        img = Image.open(file).convert('RGB')

        # Resize image
        img = img.resize((150, 150))

        # Convert image to numpy array
        img_array = np.array(img)

        # Convert to float32 and normalize
        img_array = img_array.astype(np.float32) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # =========================
        # MODEL PREDICTION
        # =========================

        infer = model.signatures["serving_default"]

        prediction = infer(
            tf.constant(img_array)
        )

        output = list(prediction.values())[0]

        probability = float(output.numpy()[0][0])

        # =========================
        # RESULT
        # =========================

        if probability > 0.5:

            result = "PNEUMONIA"

            confidence = round(probability * 100, 2)

        else:

            result = "NORMAL"

            confidence = round((1 - probability) * 100, 2)

        # =========================
        # RETURN RESPONSE
        # =========================

        return jsonify({
            'prediction': result,
            'confidence': confidence
        })

    except Exception as e:

        print("Prediction Error:", e)

        return jsonify({
            'error': str(e)
        })

# =========================
# RUN APP
# =========================

if __name__ == '__main__':
    app.run(debug=True, port=5000)