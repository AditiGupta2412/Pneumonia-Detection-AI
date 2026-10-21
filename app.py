from datetime import datetime
from io import BytesIO

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
    overlay_heatmap,
    preprocess_image,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
MODEL_PATH = "saved_pneumonia_model"

try:
    print(" Loading Pneumonia Detection Model...")

    model = tf.saved_model.load(MODEL_PATH)

    print(" Model loaded successfully!")
    print(" AI System Ready!\n")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

gradcam_explainer = None
if model is not None:
    try:
        gradcam_explainer = GradCAMExplainer(model)
    except Exception:
        app.logger.exception(
            "Grad-CAM is unavailable; prediction will continue to work."
        )


def _read_uploaded_image(file):
    with Image.open(file.stream) as uploaded_image:
        uploaded_image.verify()

    file.stream.seek(0)
    with Image.open(file.stream) as uploaded_image:
        image = uploaded_image.convert('RGB')

    return image, preprocess_image(image)


@app.route('/')
def home():
    return render_template('index.html')


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({'error': 'Image is too large. Please upload an image under 10 MB.'}), 413


@app.route('/predict', methods=['POST'])
def predict():

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
        infer = model.signatures["serving_default"]

        prediction = infer(
            tf.constant(img_array)
        )

        output = list(prediction.values())[0]

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

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            'error': 'The uploaded file is not a valid image. Please try a PNG or JPEG X-ray.'
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            'error': 'Prediction failed. Please try again.'
        }), 500


@app.route('/explain', methods=['POST'])
def explain():
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

        display_image = original_image.copy()
        display_image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        raw_heatmap = generate_raw_heatmap(
            gradcam_result.heatmap,
            display_image.size,
        )
        overlay = overlay_heatmap(display_image, gradcam_result.heatmap)

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

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            'error': 'The uploaded file is not a valid image. Please try a PNG or JPEG X-ray.'
        }), 400

    except Exception:
        app.logger.exception("Grad-CAM generation failed")
        return jsonify({
            'error': 'The explanation could not be generated. Your prediction is still available.'
        }), 500


@app.route('/report', methods=['POST'])
def report():
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

        display_image = original_image.copy()
        display_image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        raw_heatmap = generate_raw_heatmap(result.heatmap, display_image.size)
        overlay = overlay_heatmap(display_image, result.heatmap)

        original_buffer = BytesIO()
        heatmap_buffer = BytesIO()
        overlay_buffer = BytesIO()
        original_image.save(original_buffer, format='PNG')
        raw_heatmap.save(heatmap_buffer, format='PNG')
        overlay.save(overlay_buffer, format='PNG')
        original_buffer.seek(0)
        heatmap_buffer.seek(0)
        overlay_buffer.seek(0)

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

        score = result.confidence * 100
        story = [
            Paragraph('Chest X-ray Model Analysis', styles['ReportTitle']),
            Paragraph(
                f"Generated {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
                styles['ReportBody'],
            ),
            Spacer(1, 14),
        ]

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

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='chest-xray-analysis-report.pdf',
        )

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            'error': 'The uploaded file is not a valid image. Please try a PNG or JPEG X-ray.'
        }), 400

    except Exception:
        app.logger.exception("PDF report generation failed")
        return jsonify({
            'error': 'The PDF report could not be generated. Please try again.'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)