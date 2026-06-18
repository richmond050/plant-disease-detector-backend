import os
import gdown
import tensorflow as tf

MODEL_PATH = "models/folia_2.keras"
MODEL_URL = "https://drive.google.com/uc?id=1L3BvfYpPrhVVfu8kuiGSnqNgzLe-b5LL"
TFLITE_PATH = "models/folia_2.tflite"

def convert():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    
    print("Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    print("Saving TFLite model...")
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print("✅ Model converted and saved to", TFLITE_PATH)

if __name__ == "__main__":
    convert()
