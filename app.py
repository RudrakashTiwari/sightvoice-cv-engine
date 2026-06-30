"""Flask API for the CV Perception Engine.

POST /analyze   multipart: image (file), language (optional), min_confidence (optional)
GET  /health    liveness check
"""
from flask import Flask, request, jsonify
from perception.pipeline import Pipeline
import config

app = Flask(__name__)
pipeline = Pipeline()      # models load once here, at startup

@app.route("/")
def home():
    return jsonify({
        "message": "CV Perception Engine API",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({
            "status": "error", "code": "NO_IMAGE",
            "message": "No image file uploaded under field 'image'.",
        }), 400

    image_bytes = request.files["image"].read()
    if not image_bytes:
        return jsonify({
            "status": "error", "code": "EMPTY_IMAGE",
            "message": "Uploaded image is empty.",
        }), 400

    language = request.form.get("language", config.DEFAULT_LANGUAGE)
    min_conf = request.form.get("min_confidence", type=float)

    try:
        result = pipeline.run(image_bytes, language=language, min_conf=min_conf)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error", "code": "ANALYZE_FAILED", "message": str(e),
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
