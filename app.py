from flask import Flask, render_template, request, redirect, send_from_directory, url_for, jsonify
import numpy as np
import json
import uuid
import os
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

with open("plant_disease.json", 'r', encoding='utf-8') as file:
    plant_disease = json.load(file)


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html', diseases=plant_disease)


def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature


def model_predict(image):
    """Run prediction and return disease info with confidence score."""
    img = extract_features(image)
    prediction = model.predict(img)
    confidence = float(np.max(prediction)) * 100
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label, round(confidence, 1)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """AJAX API endpoint — returns JSON instead of a full page render."""
    if 'img' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    image = request.files['img']

    if image.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(image.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or WEBP.'}), 400

    # Check file size
    image.seek(0, os.SEEK_END)
    file_size = image.tell()
    image.seek(0)
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': 'File too large. Maximum size is 10 MB.'}), 400

    try:
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        filepath = f'{temp_name}_{image.filename}'
        image.save(filepath)

        prediction, confidence = model_predict(f'./{filepath}')

        return jsonify({
            'success': True,
            'imagepath': f'/{filepath}',
            'prediction': prediction,
            'confidence': confidence,
            'is_healthy': prediction.get('severity', '') == 'none'
        })

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    """Legacy form-based upload — still works as fallback."""
    if request.method == "POST":
        if 'img' not in request.files:
            return redirect('/')

        image = request.files['img']

        if image.filename == '' or not allowed_file(image.filename):
            return redirect('/')

        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        filepath = f'{temp_name}_{image.filename}'
        image.save(filepath)

        prediction, confidence = model_predict(f'./{filepath}')

        return render_template(
            'home.html',
            result=True,
            imagepath=f'/{filepath}',
            prediction=prediction,
            confidence=confidence,
            diseases=plant_disease
        )

    else:
        return redirect('/')


if __name__ == "__main__":
    app.run(debug=True)