import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)

# ── 1. SILENCE TF SPAM ────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# ── 2. LIMIT THREADS (critical for 512MB Render free tier) ───────────────────
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ── 3. PREVENT TF GRABBING ALL RAM AT ONCE ───────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ── 4. FLASK APP ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response

# ── 5. LOAD MODEL ONCE AT STARTUP ────────────────────────────────────────────
print("[CNN-VIZ] Loading MobileNetV2...", flush=True)
model = MobileNetV2(weights="imagenet")
# Warm-up pass: forces TF to compile graph now, not on first user request
model.predict(np.zeros((1, 224, 224, 3)), verbose=0)
print("[CNN-VIZ] Model ready.", flush=True)

# ── 6. EXPLANATION GENERATOR ─────────────────────────────────────────────────
def generate_explanation(pred_label):
    label = pred_label.lower().replace("_", " ")
    dog_kw = ["ridgeback","vizsla","retriever","terrier","spaniel","husky",
              "hound","dog","corgi","pug","shepherd","beagle","poodle",
              "saluki","weimaraner","redbone","labrador","dingo","malinois",
              "boxer","bulldog","dalmatian","doberman","rottweiler","setter"]
    cat_kw = ["cat","tabby","tiger","panther","leopard","siamese","persian",
              "egyptian","burmese","manx","angora","ragdoll","cheetah","cougar"]
    if any(k in label for k in dog_kw):
        return f"Detected {label.upper()} due to edge structures (ears, snout), fur texture patterns, and facial geometry."
    elif any(k in label for k in cat_kw):
        return f"Detected {label.upper()} due to sharp ear edges, whisker patterns, and compact facial features."
    return f"Classified as {label.upper()} based on hierarchical feature extraction across convolution and dense layers."

# ── 7. PREDICT ROUTE ─────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    try:
        file = request.files["image"]
        img  = Image.open(file.stream).convert("RGB").resize((224, 224))
        arr  = preprocess_input(np.expand_dims(np.array(img), axis=0).astype("float32"))
        preds   = model.predict(arr, verbose=0)
        decoded = decode_predictions(preds, top=5)[0]
        results = [{"label": lbl, "probability": float(prob)} for (_, lbl, prob) in decoded]
        return jsonify({
            "predictions": results,
            "explanation": generate_explanation(results[0]["label"])
        })
    except Exception as e:
        print(f"[CNN-VIZ] Predict error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

# ── 8. HEALTH CHECK ───────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# ── 9. SERVE FRONTEND ─────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    try:
        with open("index.html", "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return "Backend is live. Deploy index.html alongside app.py."

# ── 10. ENTRY POINT — reads $PORT from Render environment ────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
