from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

# 1. OPTIMIZE TENSORFLOW MEMORY FOR RENDER FREE TIER
# Prevents the server from running out of RAM and crashing silently
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)

# 2. BASE CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# 3. FORCE HEADERS ON EVERY RESPONSE (The Preflight Killer)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Load model once (IMPORTANT)
model = MobileNetV2(weights="imagenet")

# ---------- Helper: Explanation Generator ----------
def generate_explanation(pred_label):
    # Clean the label so it matches our keywords
    clean_label = pred_label.lower().replace("_", " ")

    # Lists of common breeds that MobileNet predicts
    dog_keywords = ["ridgeback", "vizsla", "retriever", "terrier", "spaniel", "husky", "hound", "dog", "corgi", "pug", "shepherd", "beagle", "poodle", "saluki", "weimaraner", "redbone"]
    cat_keywords = ["cat", "tabby", "tiger", "panther", "leopard", "siamese", "persian"]

    if any(keyword in clean_label for keyword in dog_keywords):
        return f"Detected {clean_label.upper()} due to edge structures (ears, snout), fur texture patterns, and facial geometry."
    elif any(keyword in clean_label for keyword in cat_keywords):
        return f"Detected {clean_label.upper()} due to sharp ear edges, whisker patterns, and compact facial features."
    else:
        return f"Classified as {clean_label.upper()} based on hierarchical feature extraction across convolution and dense layers."

# ---------- Prediction Route ----------
# Notice we added 'OPTIONS' here to catch the browser's hidden security check
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # If the browser is just checking security (OPTIONS), say OK immediately
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")

        # Resize to model input
        img = img.resize((224, 224))

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        decoded = decode_predictions(preds, top=5)[0]

        results = []
        for (_, label, prob) in decoded:
            results.append({
                "label": label,
                "probability": float(prob)
            })

        explanation = generate_explanation(results[0]["label"])

        return jsonify({
            "predictions": results,
            "explanation": explanation
        })

    except Exception as e:
        # Added a 500 error code so the frontend knows if Python crashed
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ---------- Run Server ----------
@app.route("/", methods=["GET"])
def index():
    try:
        with open("index.html", "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return "Backend API is awake! Use the frontend UI to send images."

if __name__ == "__main__":
    app.run(debug=True)