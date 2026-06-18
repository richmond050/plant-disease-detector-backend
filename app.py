from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import numpy as np
import json
from PIL import Image

# Try importing tflite-runtime, fallback to tensorflow.lite if running locally
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        # Fallback if running on local development machine with full TF
        import tensorflow.lite as tflite

app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

MODEL_PATH = "models/folia_2.tflite"
MODEL_INFO_PATH = "models/folia_2_model_info.json"

# Load model and model info
def load_model_and_info():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"TFLite model file not found at {MODEL_PATH}. Make sure it is committed to Git.")
    
    # Load the TFLite model interpreter
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
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
    
    return interpreter, class_names

# Load model and class names
model, class_names = load_model_and_info()
print(f"✅ TFLite Model loaded successfully from {MODEL_PATH}")
print(f"✅ Number of classes: {len(class_names)}")

# Function to clean class name
def clean_class_name(raw_name):
    name = raw_name.replace("___", " ").replace("_", " ")
    words = name.split()
    cleaned = []
    for word in words:
        if not cleaned or word.lower() != cleaned[-1].lower():
            cleaned.append(word)
    return " ".join(word.capitalize() for word in cleaned)

# Prediction function using TFLite
def predict_image(img_path):
    # Load and resize image using PIL
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    
    # Preprocess image
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get input and output tensors details
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    # Point the model to the image array
    model.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    model.invoke()
    
    # Extract the results
    prediction = model.get_tensor(output_details[0]['index'])
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
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)