# CNN.VISUALIZER — Interactive Deep Learning Architecture Explorer

> **Live Demo →** [cnn-visualizer-ffaj.onrender.com](https://cnn-visualizer-ffaj.onrender.com)  
> **Stack:** Python · Flask · TensorFlow · MobileNetV2 · HTML5 Canvas · Vanilla JS · Render

---

## What Is This?

CNN.VISUALIZER is a full-stack interactive tool that lets you **see inside a real Convolutional Neural Network** as it processes your image — layer by layer, in real time.

Upload any photo. Watch the signal travel through 9 architectural stages — from raw pixels to class probabilities — with animated 3D layer visualization, live feature maps, and a detailed inspector panel for every operation. The backend runs **MobileNetV2 trained on ImageNet** and returns top-5 predictions with confidence scores.

This wasn't built to be a tutorial. It was built to answer one question I kept asking myself when studying CNNs: *"What is the network actually seeing right now?"*

---

## Why I Built This

When I started studying deep learning, parameter calculations and layer-by-layer reasoning felt abstract and hard to hold in my head. I would see the equations, understand them individually, then lose the thread when trying to connect them across a full architecture. I kept thinking — why isn't there something I can just *look at* while the math is happening?

So I built it.

---

## Features

- **Real AI inference** — MobileNetV2 (ImageNet weights) runs on every uploaded image via Flask REST API
- **9-layer architecture walkthrough** — Input → Conv1 → ReLU6 → Depthwise → Pointwise → Bottleneck → Global Pool → FC → Softmax
- **3D isometric layer visualization** — animated signal-flow particles travel between layers on an HTML5 Canvas
- **Live feature maps** — 8 client-side pixel transforms simulate what each filter type detects (edges, gradients, colour channels, thresholds)
- **Layer Inspector** — operation formula, parameter count, FLOPs, memory, real-world analogy, and interview prep insight for every layer
- **Class probability bars** — animated softmax output showing top-5 predictions with confidence percentages
- **Low-confidence warning** — automatically flags predictions below 30% and explains common failure modes
- **X-Ray mode** — hover any feature map to zoom 3.2× inline without leaving the page
- **Auto-retry on cold start** — 3x automatic retry with 8-second gaps handles Render free-tier spin-up transparently
- **Backend health indicator** — live green/red status badge pings `/health` every 15 seconds
- **Responsive layout** — side panels collapse on narrow viewports so the canvas stays usable
- **Keyboard navigation** — Arrow keys step through layers, Space bar plays/pauses the forward pass animation

---

## Architecture

```
Browser (Frontend)
├── HTML5 Canvas       3D layer rendering, particles, kernel animation
├── Vanilla JS         State machine, API integration, feature map engine
└── CSS Grid           3-column layout with responsive breakpoints

        ↕  POST /predict  (multipart/form-data)

Flask Backend (Render)
├── POST /predict      PIL resize → preprocess_input → MobileNetV2.predict()
│                      → decode_predictions(top=5) → JSON response
├── GET  /health       { "status": "ok" }  (lightweight ping)
└── GET  /             Serves index.html (single-service deployment)
```

### CNN Layer Reference

| # | Layer | Type | Output Shape | Params |
|---|-------|------|-------------|--------|
| 0 | Input | — | 3×224×224 | 0 |
| 1 | Conv 1 | Conv2D + BN | 32×112×112 | 864 |
| 2 | ReLU6 | Activation | 32×112×112 | 0 |
| 3 | Depthwise | Depthwise Conv2D | 32×112×112 | 288 |
| 4 | Pointwise | 1×1 Conv2D | 16×112×112 | 512 |
| 5 | Bottleneck | Inverted Residual Block | 24×56×56 | 3,432 |
| 6 | Global Pool | Global Average Pooling | 1280 | 0 |
| 7 | FC / Dense | Linear | 1000 | 1,281,000 |
| 8 | Softmax | — | 1000 | 0 |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5 Canvas, Vanilla JS (ES2022), CSS Grid |
| Backend | Python 3.11, Flask 3.0, Flask-CORS |
| ML Model | TensorFlow 2.15 (CPU), MobileNetV2, ImageNet |
| Image Processing | Pillow, NumPy |
| Deployment | Render (Web Service), Gunicorn |
| Fonts | Orbitron, JetBrains Mono |

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/Tejasvi-eng/CNN-Visualizer.git
cd CNN-Visualizer

# 2. Install (no GPU required — uses tensorflow-cpu)
pip install -r requirements.txt

# 3. Run
python app.py
# Open http://localhost:5000
```

---

## Deployment (Render)

Single Web Service — Flask serves both the frontend and the prediction API.

```
Build command:  pip install -r requirements.txt
Start command:  gunicorn app:app
Health check:   /health
Python version: 3.11.0  (PYTHON_VERSION env var)
```

**Free-tier cold start:** Server sleeps after 15 minutes of inactivity. First request after sleep takes 30–60 seconds (TensorFlow model loading). The frontend handles this with a silent 3x retry loop — users see a "waking up" toast instead of an error.

---

## Image Upload Requirements

| Property | Limit |
|----------|-------|
| Formats | JPEG, PNG, WebP |
| Max size | 10 MB |
| Processing | Auto-resized to 224×224 |
| Best results | Single centred subject, clear background |

---

## Known Limitations

- Feature maps are **client-side simulations** (pixel transforms), not actual CNN activations from the model. Real Grad-CAM heatmaps are on the roadmap.
- The walkthrough shows 9 representative stages. BatchNorm layers are folded into each conv description — this is intentional for clarity.
- Free-tier cold starts can take up to 60 seconds on first daily use.

---

## File Structure

```
CNN-Visualizer/
├── app.py            Flask server — /predict, /health, serves index.html
├── index.html        Frontend — canvas engine, UI, API integration
├── requirements.txt  Python dependencies
├── render.yaml       Render deployment config
└── README.md
```

---

## Roadmap

- [ ] Real Grad-CAM activation heatmaps
- [ ] Side-by-side image comparison mode
- [ ] Additional architecture support (ResNet50, EfficientNet)
- [ ] Export feature maps as PNG
- [ ] Mobile layout optimisation

---

## License

MIT

---

*Built by [Tejasvi](https://github.com/Tejasvi-eng)*
