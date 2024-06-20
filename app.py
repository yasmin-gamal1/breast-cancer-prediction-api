import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
import os
from PIL import Image  # Import PIL to verify the image

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model('my_model.h5')

def preprocess_image(image_file):
    try:
        # Verify the image
        img = Image.open(io.BytesIO(image_file.read()))
        img.verify()  # This will raise an exception if the file is not an image

        # Rewind the file pointer
        image_file.seek(0)

        # Read the file into a byte stream and load the image from it
        img = image.load_img(io.BytesIO(image_file.read()), target_size=(50, 50))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        raise ValueError("Please enter a valid breast cancer image")

@app.route('/')
def home():
    return "Welcome to the Breast Cancer Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    print("Request Headers:", request.headers)
    print("Request Form:", request.form)
    print("Request Files:", request.files)
    
    # Print all file keys received
    for file_key in request.files:
        print(f"File key: {file_key}")
    
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    
    # Additional debug info
    print(f"Received file: {file.filename}")
    
    if file.filename == '':
        print("No file selected")
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        img = preprocess_image(file)
        prediction = model.predict(img)
        pred_label = np.argmax(prediction, axis=1)[0]
        print(f"Prediction: {pred_label}")
        return jsonify({'prediction': int(pred_label)})
    except ValueError as ve:
        print(f"Error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
