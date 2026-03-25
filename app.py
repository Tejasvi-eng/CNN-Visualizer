from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

app = Flask(__name__)
CORS(app)

# Load model once (IMPORTANT)
model = MobileNetV2(weights="imagenet")

# ---------- Helper: Explanation Generator ----------
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
@app.route("/predict", methods=["POST"])
def predict():
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
        return jsonify({"error": str(e)})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ---------- Run Server ----------
@app.route("/", methods=["GET"])
def index():
    with open("index.html", "r", encoding="utf-8", errors="replace") as f:
        return f.read()

if __name__ == "__main__":
    app.run(debug=True)