"""Flask web application for the Pneumonia Detection AI research demo.

Routes:
    GET  /         the single-page interface (templates/index.html)
    POST /predict  classify an uploaded chest X-ray as NORMAL or PNEUMONIA
    POST /explain  return a Grad-CAM heat-map and overlay for the prediction
    POST /report   return a PDF report with the prediction and Grad-CAM images

Every POST route receives the image in the form field "file", processes it in
memory and stores nothing on the server. Errors are returned as JSON objects
with an "error" field that the web page shows to the user.
"""

import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
from PIL import UnidentifiedImageError
import numpy as np
import tensorflow as tf
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from gradcam import (
    GradCAMExplainer,
    generate_raw_heatmap,
    image_to_data_url,
    normalize_to_rgb,
    overlay_heatmap,
    preprocess_image,
)

app = Flask(__name__)
# Flask rejects larger request bodies with HTTP 413 before a route runs.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
# Resolved from this file's location so the app works from any working folder.
MODEL_PATH = Path(__file__).resolve().parent / "saved_pneumonia_model"
# Pillow format names; checked on the server as well as in the browser.
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG"}
# Large chest X-rays are around 4300x4300 (18.5M pixels).
MAX_IMAGE_PIXELS = 50_000_000
INVALID_IMAGE_MESSAGE = 'The uploaded file is not a valid image. Please try a PNG or JPEG X-ray.'


class InvalidImageError(ValueError):
    """Raised when an upload is not an image the app can analyze."""

    def __init__(self, message=INVALID_IMAGE_MESSAGE):
        super().__init__(message)


# Load the exported VGG16 SavedModel once at start-up. If loading fails the
# server still starts, and the routes answer with HTTP 503 instead of crashing.
try:
    print(" Loading Pneumonia Detection Model...")

    model = tf.saved_model.load(str(MODEL_PATH))

    print(" Model loaded successfully!")
    print(" AI System Ready!\n")

except Exception as e:
    # Plain ASCII: an emoji here crashes on Windows consoles that use cp1252.
    print(f" Error loading model: {e}")
    model = None

# Grad-CAM is optional: the explainer verifies that its rebuilt model matches
# the SavedModel, and if that check fails only /explain and /report are
# disabled while /predict keeps working.
gradcam_explainer = None
if model is not None:
    try:
        gradcam_explainer = GradCAMExplainer(model)
    except Exception:
        app.logger.exception(
            "Grad-CAM is unavailable; prediction will continue to work."
        )


def _read_uploaded_image(file):
    """Validate an uploaded file and prepare it for the model.

    Returns the image as 8-bit RGB (for display and the PDF) and the
    pre-processed batch of shape (1, 150, 150, 3) for the model. Raises
    InvalidImageError, which the routes turn into HTTP 400, for any problem
    with the file itself.
    """
    try:
        # First pass: Image.open only reads the header, so the format and size
        # are checked before any pixels are decoded. verify() then checks the
        # file structure for corruption.
        with Image.open(file.stream) as uploaded_image:
            if uploaded_image.format not in ALLOWED_IMAGE_FORMATS:
                raise InvalidImageError('Unsupported image format. Please upload a PNG or JPEG X-ray.')
            width, height = uploaded_image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageError('Image dimensions are too large. Please upload a smaller X-ray.')
            uploaded_image.verify()

        # Second pass: an image cannot be used after verify(), so the stream
        # is rewound and the image opened again to decode the pixels.
        file.stream.seek(0)
        with Image.open(file.stream) as uploaded_image:
            image = normalize_to_rgb(uploaded_image)

    except InvalidImageError:
        raise
    except Image.DecompressionBombError as error:
        # Raised by Pillow for images far beyond MAX_IMAGE_PIXELS.
        raise InvalidImageError('Image dimensions are too large. Please upload a smaller X-ray.') from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        # Pillow reports corrupt files with any of these exception types.
        raise InvalidImageError() from error

    return image, preprocess_image(image)


@app.route('/')
def home():
    return render_template('index.html')


@app.errorhandler(413)
def file_too_large(_error):
    """Return the 10 MB limit error as JSON so the page can display it."""
    return jsonify({'error': 'Image is too large. Please upload an image under 10 MB.'}), 413


@app.route('/predict', methods=['POST'])
def predict():
    """Classify the uploaded X-ray and return the class and score as JSON."""

    # Check model loaded
    if model is None:
        return jsonify({
            'error': 'Model failed to load.'
        }), 503

    # Check file uploaded
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file uploaded.'
        }), 400

    file = request.files['file']

    # Check file selected
    if file.filename == '':
        return jsonify({
            'error': 'No file selected.'
        }), 400

    try:
        _, img_array = _read_uploaded_image(file)
        # The SavedModel's serving signature returns a dict holding one
        # (1, 1) tensor: the sigmoid score for PNEUMONIA.
        infer = model.signatures["serving_default"]

        prediction = infer(
            tf.constant(img_array)
        )

        output = list(prediction.values())[0]

        # Scores above 0.5 mean PNEUMONIA. For NORMAL the shown value is
        # 1 - score, so it always describes the predicted class. It is the
        # model's raw output, not a calibrated probability.
        probability = float(output.numpy()[0][0])
        if probability > 0.5:

            result = "PNEUMONIA"

        else:

            result = "NORMAL"

            probability = 1 - probability
        confidence = round(probability * 100, 2)
        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'score': confidence
        })

    except InvalidImageError as error:
        # A problem with the uploaded file: the user can fix it.
        return jsonify({'error': str(error)}), 400

    except Exception:
        # Anything else is a server-side failure, including model errors.
        app.logger.exception("Prediction failed")

        return jsonify({
            'error': 'Prediction failed. Please try again.'
        }), 500


@app.route('/explain', methods=['POST'])
def explain():
    """Return the Grad-CAM heat-map and overlay for the uploaded X-ray as JSON.

    The images are sent as PNG data URLs so the page can show them directly.
    """
    if model is None:
        return jsonify({'error': 'The prediction model is unavailable.'}), 503

    if gradcam_explainer is None:
        return jsonify({
            'error': 'An explanation could not be generated for this model. The prediction is still available.'
        }), 503

    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'Please select a chest X-ray image first.'}), 400

    file = request.files['file']
    try:
        original_image, image_array = _read_uploaded_image(file)
        gradcam_result = gradcam_explainer.generate(image_array)

        # The heat-map is a coarse 9 x 9 grid. It is scaled up to a copy of
        # the X-ray that is at most 1200 px on its longer side, which keeps
        # the response small.
        display_image = original_image.copy()
        display_image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        raw_heatmap = generate_raw_heatmap(
            gradcam_result.heatmap,
            display_image.size,
        )
        overlay = overlay_heatmap(display_image, gradcam_result.heatmap)

        # Diagnostics go to the server log (and to the "debug" field below)
        # so a heat-map can be traced back to the exact model output.
        app.logger.info(
            "Grad-CAM diagnostics: predicted_class=%s raw_model_output=%.8f "
            "confidence=%.4f target_class=%s target_index=%s target_logit=%.8f layer=%s "
            "feature_map_shape=%s gradient_shape=%s heatmap_min=%.6f "
            "heatmap_max=%.6f",
            gradcam_result.predicted_class,
            gradcam_result.pneumonia_probability,
            gradcam_result.confidence * 100,
            gradcam_result.target_class,
            gradcam_result.target_index,
            gradcam_result.target_logit,
            gradcam_explainer.target_layer_name,
            gradcam_result.feature_map_shape,
            gradcam_result.gradient_shape,
            gradcam_result.heatmap_min,
            gradcam_result.heatmap_max,
        )
        return jsonify({
            'prediction': gradcam_result.predicted_class,
            'confidence': round(gradcam_result.confidence * 100, 2),
            'target_layer': gradcam_explainer.target_layer_name,
            'raw_gradcam': image_to_data_url(raw_heatmap),
            'overlay': image_to_data_url(overlay),
            'debug': {
                'raw_model_output': gradcam_result.pneumonia_probability,
                'target_class': gradcam_result.target_class,
                'target_index': gradcam_result.target_index,
                'target_logit': gradcam_result.target_logit,
                'feature_map_shape': gradcam_result.feature_map_shape,
                'gradient_shape': gradcam_result.gradient_shape,
                'heatmap_min': gradcam_result.heatmap_min,
                'heatmap_max': gradcam_result.heatmap_max,
            }
        })

    except InvalidImageError as error:
        return jsonify({'error': str(error)}), 400

    except Exception:
        app.logger.exception("Grad-CAM generation failed")
        return jsonify({
            'error': 'The explanation could not be generated. Your prediction is still available.'
        }), 500


@app.route('/report', methods=['POST'])
def report():
    """Build a one-page PDF report for the uploaded X-ray and send it as a download.

    The report shows the prediction, the model score, the X-ray, the raw
    Grad-CAM map and the overlay, and a research-use disclaimer. It is built
    entirely in memory.
    """
    if model is None:
        return jsonify({'error': 'The prediction model is unavailable.'}), 503

    if gradcam_explainer is None:
        return jsonify({
            'error': 'A report with Grad-CAM could not be generated for this model.'
        }), 503

    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'Please select a chest X-ray image first.'}), 400

    file = request.files['file']
    try:
        original_image, image_array = _read_uploaded_image(file)
        result = gradcam_explainer.generate(image_array)

        # The resized copy (not the full-size upload) goes into the PDF so the
        # report stays small even for very large X-rays.
        display_image = original_image.copy()
        display_image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        raw_heatmap = generate_raw_heatmap(result.heatmap, display_image.size)
        overlay = overlay_heatmap(display_image, result.heatmap)

        # ReportLab reads images from file-like objects, so each image is
        # saved as PNG into an in-memory buffer.
        original_buffer = BytesIO()
        heatmap_buffer = BytesIO()
        overlay_buffer = BytesIO()
        display_image.save(original_buffer, format='PNG')
        raw_heatmap.save(heatmap_buffer, format='PNG')
        overlay.save(overlay_buffer, format='PNG')
        original_buffer.seek(0)
        heatmap_buffer.seek(0)
        overlay_buffer.seek(0)

        # Page layout and text styles for the report.
        pdf_buffer = BytesIO()
        document = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=34,
            bottomMargin=34,
            title='Chest X-ray Model Analysis Report',
            author='Pneumonia Detection AI Research Demo',
        )
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=styles['Title'],
            alignment=TA_LEFT,
            textColor=colors.HexColor('#142b49'),
            fontSize=20,
            leading=24,
            spaceAfter=5,
        ))
        styles.add(ParagraphStyle(
            name='ReportBody',
            parent=styles['BodyText'],
            textColor=colors.HexColor('#40536b'),
            fontSize=9,
            leading=13,
        ))
        styles.add(ParagraphStyle(
            name='ReportCaption',
            parent=styles['BodyText'],
            alignment=TA_CENTER,
            textColor=colors.HexColor('#40536b'),
            fontSize=8,
            leading=10,
        ))
        styles.add(ParagraphStyle(
            name='ReportDisclaimer',
            parent=styles['BodyText'],
            textColor=colors.HexColor('#67542a'),
            fontSize=8,
            leading=11,
        ))

        # The "story" is ReportLab's ordered list of page elements: title,
        # summary table, image row, explanation and disclaimer.
        score = result.confidence * 100
        story = [
            Paragraph('Chest X-ray Model Analysis', styles['ReportTitle']),
            Paragraph(
                f"Generated {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
                styles['ReportBody'],
            ),
            Spacer(1, 14),
        ]

        # Summary row: predicted class, score and the class Grad-CAM explains.
        summary = Table(
            [
                ['MODEL PREDICTION', 'MODEL SCORE', 'EXPLAINED CLASS'],
                [
                    result.predicted_class,
                    f'{score:.2f}%',
                    result.target_class,
                ],
            ],
            colWidths=[180, 180, 180],
        )
        summary.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf4fc')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#315d8b')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#142b49')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, 1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#d7e2ef')),
        ]))
        story.extend([
            summary,
            Spacer(1, 8),
            Paragraph(
                'The model score is not calibrated diagnostic certainty or measured accuracy.',
                styles['ReportBody'],
            ),
            Spacer(1, 16),
        ])

        # Three images side by side with captions underneath. kind='proportional'
        # fits each image inside the box without stretching it.
        image_width = 170
        image_height = 190
        images = [
            ReportImage(original_buffer, width=image_width, height=image_height, kind='proportional'),
            ReportImage(heatmap_buffer, width=image_width, height=image_height, kind='proportional'),
            ReportImage(overlay_buffer, width=image_width, height=image_height, kind='proportional'),
        ]
        captions = [
            Paragraph('Original X-ray', styles['ReportCaption']),
            Paragraph('Raw Grad-CAM', styles['ReportCaption']),
            Paragraph('Grad-CAM Overlay', styles['ReportCaption']),
        ]
        image_table = Table(
            [images, captions],
            colWidths=[180, 180, 180],
            rowHeights=[200, 22],
        )
        image_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f7f9fc')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e3eaf2')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e3eaf2')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.extend([
            image_table,
            Spacer(1, 16),
            Paragraph(
                f'Grad-CAM highlights relative image regions that influenced the '
                f'{result.target_class} prediction. It is a coarse attribution map, '
                'not a probability map, lesion boundary, or diagnosis.',
                styles['ReportBody'],
            ),
            Spacer(1, 12),
            Paragraph(
                '<b>Research use only.</b> This report is generated by an AI demo, '
                'not a medical device. Do not use it to diagnose, treat, or rule out '
                'pneumonia. Consult a qualified healthcare professional.',
                styles['ReportDisclaimer'],
            ),
        ])
        document.build(story)
        pdf_buffer.seek(0)

        # as_attachment makes the browser download the file instead of opening it.
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='chest-xray-analysis-report.pdf',
        )

    except InvalidImageError as error:
        return jsonify({'error': str(error)}), 400

    except Exception:
        app.logger.exception("PDF report generation failed")
        return jsonify({
            'error': 'The PDF report could not be generated. Please try again.'
        }), 500


if __name__ == '__main__':
    # The Werkzeug debugger allows running code, so it is opt-in: FLASK_DEBUG=1.
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', port=5000)