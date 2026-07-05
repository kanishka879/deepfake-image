"""
Deepfake Detector — Backend API
--------------------------------
This is a REST API server. Its only job is:
  1. Receive an image from the frontend (the HTML/JS file)
  2. Run it through YOUR model
  3. Send back a JSON answer: { "label": "real" | "fake", "confidence": 0.0-1.0 }

Wherever your trained model goes, look for the "PLUG YOUR MODEL IN HERE" comments.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import io

# =========================================================
# 1. APP SETUP
# =========================================================
app = Flask(__name__)

# CORS = Cross-Origin Resource Sharing.
# Your HTML file and this server are considered different "origins"
# (different ports, or one is a local file). Browsers block requests
# between different origins by default for security. This line tells
# the browser "it's fine, let the frontend talk to me."
CORS(app)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMAGE_SIZE = 224  # MUST match the size used in train.py
MODEL_PATH = "deepfake_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must be identical to the normalization used in train.py — the model was
# trained expecting inputs in this exact numeric range/distribution.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================================================
# 2. LOAD THE MODEL ONCE, AT STARTUP
# =========================================================
# Loading a model is slow (can take seconds). We do it ONE time when the
# server starts, and keep it in memory — NOT inside the /predict function.
# If you loaded it per-request, every single API call would be slow.

def load_model():
    # Rebuild the EXACT same architecture used in train.py. A .pt file
    # (from torch.save(model.state_dict(), ...)) only stores weight
    # numbers, not the architecture itself — so we must reconstruct the
    # same skeleton before the weights can be loaded into it.
    architecture = models.resnet18(weights=None)  # None: we're loading our own weights, not ImageNet's
    num_features = architecture.fc.in_features
    architecture.fc = nn.Linear(num_features, 1)

    architecture.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    architecture.to(DEVICE)
    architecture.eval()  # inference mode: disables dropout/batchnorm updates
    return architecture


model = load_model()

# Same preprocessing pipeline as val_transform in train.py.
# Inference preprocessing must match training preprocessing exactly.
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def allowed_file(filename):
    """Checks the file extension is one we accept."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes):
    """
    Turns raw uploaded bytes into a tensor shaped exactly like what the
    model saw during training: resized, converted to a tensor, normalized,
    and given a "batch" dimension (models expect a batch even if it's 1 image).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = inference_transform(image)          # shape: (3, 224, 224)
    tensor = tensor.unsqueeze(0)                 # shape: (1, 3, 224, 224) — batch of 1
    return tensor.to(DEVICE)


def run_inference(input_tensor):
    """
    Runs the model and returns (label, confidence).

    torch.no_grad(): we're not training, so skip gradient tracking —
    faster and uses less memory during inference.
    """
    with torch.no_grad():
        output = model(input_tensor)              # raw score ("logit")
        probability = torch.sigmoid(output).item()  # squashed to 0-1

    # Recall from train.py: class_to_idx maps folder names alphabetically,
    # e.g. {'fake': 0, 'real': 1} — so a HIGH probability means "real" here.
    # Double check your own printed class_to_idx from training and flip
    # this comparison if your mapping came out the other way around.
    label = "real" if probability > 0.5 else "fake"
    confidence = probability if label == "real" else 1 - probability
    return label, round(confidence, 4)


# =========================================================
# 4. ROUTES (the actual API endpoints)
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    """
    A simple endpoint to check the server is alive.
    Visiting http://localhost:5000/health in a browser should
    show {"status": "ok"}. Common pattern in real APIs for
    monitoring/load balancers to check a service is up.
    """
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    The main endpoint the frontend calls.
    Expects a multipart/form-data POST request with a field named "image".
    """

    # --- Step 1: validate the request ---
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # --- Step 2: run the pipeline ---
    try:
        image_bytes = file.read()
        processed = preprocess_image(image_bytes)
        label, confidence = run_inference(processed)
    except Exception as e:
        # Never let a raw crash leak to the client — return a clean 500 instead.
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # --- Step 3: respond ---
    return jsonify({"label": label, "confidence": confidence})


# =========================================================
# 5. ENTRY POINT
# =========================================================
if __name__ == "__main__":
    # debug=True auto-reloads the server when you edit code, and shows
    # detailed error pages. Turn it off (debug=False) in production.
    app.run(host="0.0.0.0", port=5000, debug=True)
