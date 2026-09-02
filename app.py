"""
Web app for the image classifier.

    python app.py
    -> open http://localhost:5000 in your browser, upload a photo, get labels.

Also exposes a JSON API for other programs:

    curl -F "image=@images/dog.jpg" http://localhost:5000/predict
    {"predictions": [{"label": "Rottweiler", "probability": 0.733}, ...]}
"""

import base64
import io

from flask import Flask, jsonify, render_template, request
from PIL import UnidentifiedImageError

from classify import classify, load_model

MAX_UPLOAD_MB = 10
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Load the model once at startup; every request reuses it.
MODEL, PREPROCESS, LABELS = load_model()


def predict_upload(file_storage):
    """Returns (predictions, error). predictions is a list of {label, probability}."""
    if file_storage is None or file_storage.filename == "":
        return None, "Please choose an image file."
    if file_storage.mimetype not in ALLOWED_TYPES:
        return None, f"Unsupported file type '{file_storage.mimetype}'. Use JPEG, PNG, WebP, GIF or BMP."
    data = file_storage.read()
    try:
        results = classify(io.BytesIO(data), MODEL, PREPROCESS, LABELS, top_k=5)
    except UnidentifiedImageError:
        return None, "That file does not look like a valid image."
    predictions = [{"label": label, "probability": round(prob, 4)} for label, prob in results]
    return predictions, None


@app.route("/", methods=["GET", "POST"])
def index():
    predictions, error, preview = None, None, None
    if request.method == "POST":
        upload = request.files.get("image")
        predictions, error = predict_upload(upload)
        if predictions:
            # Show the uploaded image back to the user without saving it to disk.
            upload.seek(0)
            encoded = base64.b64encode(upload.read()).decode("ascii")
            preview = f"data:{upload.mimetype};base64,{encoded}"
    return render_template("index.html", predictions=predictions, error=error, preview=preview)


@app.route("/predict", methods=["POST"])
def predict_api():
    predictions, error = predict_upload(request.files.get("image"))
    if error:
        return jsonify({"error": error}), 400
    return jsonify({"predictions": predictions})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large. Maximum size is {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
