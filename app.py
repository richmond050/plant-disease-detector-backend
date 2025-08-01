from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import requests
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import gdown

app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Google Drive configuration
MODEL_PATH = "models/plant_disease_detector_model.keras"
MODEL_INFO_PATH = "models/model_info.json"
MODEL_URL = "https://drive.google.com/uc?id=1LcT5g1Mprunevuj82fWZiclE_VJIlJE0"

# Load model and model info
def load_model_and_info():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download model from Google Drive if not exists
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            print("✅ Model downloaded successfully from Google Drive!")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            raise FileNotFoundError(f"Could not download model from Google Drive")
    
    # Load the model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # Load model info if available
    class_names = None
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH, 'r') as f:
            model_info = json.load(f)
            class_names = model_info.get('class_names', None)
    
    # Fallback to hardcoded class names if model_info.json is not available
    if class_names is None:
        class_names = [
            "Pepper__bell___Bacterial_spot",
            "Pepper__bell___healthy",
            "Potato___Early_blight",
            "Potato___healthy",
            "Potato___Late_blight",
            "Tomato_Bacterial_spot",
            "Tomato_Early_blight",
            "Tomato_healthy",
            "Tomato_Late_blight",
            "Tomato_Leaf_Mold",
            "Tomato_Septoria_leaf_spot",
            "Tomato_Spider_mites_Two_spotted_spider_mite",
            "Tomato__Target_Spot",
            "Tomato__Tomato_mosaic_virus",
            "Tomato__Tomato_YellowLeaf__Curl_Virus"
        ]
    
    return model, class_names

# Load model and class names
model, class_names = load_model_and_info()
print(f"✅ Model loaded successfully from {MODEL_PATH}")
print(f"�� Number of classes: {len(class_names)}")

# Function to clean class name
def clean_class_name(raw_name):
    name = raw_name.replace("___", " ").replace("_", " ")
    words = name.split()
    cleaned = []
    for word in words:
        if not cleaned or word.lower() != cleaned[-1].lower():
            cleaned.append(word)
    return " ".join(word.capitalize() for word in cleaned)

# Prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    raw_class = class_names[predicted_index]
    readable_class = clean_class_name(raw_class)
    return readable_class


# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "message": "Backend is working!",
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_count": len(class_names) if class_names else 0
    })

# POST /predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    # Save to temp folder
    temp_filename = f"{uuid.uuid4().hex}.jpg"
    temp_path = os.path.join("temp", temp_filename)
    os.makedirs("temp", exist_ok=True)
    file.save(temp_path)

    try:
        result = predict_image(temp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_path)  # Clean up after prediction

    return jsonify({"prediction": result})

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)