"""
CNN.VISUALIZER
Author: Tejasvi
GitHub: https://github.com/Tejasvi-eng/CNN-Visualizer
Built: 2026
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

# 1. OPTIMIZE TENSORFLOW FOR 512MB FREE TIER (Prevents 503 Crashes)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# 2. LAZY LOAD THE MODEL (Prevents Boot Timeouts)
model = None

def get_model():
    global model
    if model is None:
        print("-> Waking up MobileNetV2 (Downloading weights...)")
        model = MobileNetV2(weights="imagenet")
    return model

def generate_explanation(pred_label):
    clean_label = pred_label.lower().replace("_", " ")
    dog_keywords = ["ridgeback", "vizsla", "retriever", "terrier", "spaniel", "husky", "hound", "dog", "corgi", "pug", "shepherd", "beagle", "poodle", "saluki", "weimaraner", "redbone"]
    cat_keywords = ["cat", "tabby", "tiger", "panther", "leopard", "siamese", "persian"]

    if any(keyword in clean_label for keyword in dog_keywords):
        return f"Detected {clean_label.upper()} due to edge structures (ears, snout), fur texture patterns, and facial geometry."
    elif any(keyword in clean_label for keyword in cat_keywords):
        return f"Detected {clean_label.upper()} due to sharp ear edges, whisker patterns, and compact facial features."
    else:
        return f"Classified as {clean_label.upper()} based on hierarchical feature extraction across convolution and dense layers."

# 3. BULLETPROOF PREDICT ROUTINE
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        # Safely extract image to avoid 400 Bad Request
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400
            
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Wake up the model (Takes ~50s the very first time)
        current_model = get_model()
        
        preds = current_model.predict(img_array)
        decoded = decode_predictions(preds, top=5)[0]

        results = [{"label": label, "probability": float(prob)} for (_, label, prob) in decoded]
        explanation = generate_explanation(results[0]["label"])

        return jsonify({"predictions": results, "explanation": explanation})

    except Exception as e:
        print(f"CRASH: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/", methods=["GET"])
def index():
    try:
        with open("index.html", "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return "Backend API is awake! Use the frontend UI to send images."

# Include the other AI's port bindings just to be 100% safe
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
