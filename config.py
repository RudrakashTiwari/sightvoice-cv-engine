"""Central configuration for the CV Perception Engine.

Everything is a pretrained model that runs locally. No API keys, no external
services at inference time. Weights are downloaded once on first run, then cached.
"""

# --- Models (all pretrained, run locally) ---
DETECTOR_MODEL = "yolov8n.pt"                              # YOLOv8 nano (COCO)
DEPTH_MODEL    = "MiDaS_small"                             # MiDaS monocular depth
CAPTION_MODEL  = "Salesforce/blip-image-captioning-base"  # BLIP image captioning

# --- Detection ---
MIN_CONFIDENCE = 0.35
NMS_IOU        = 0.45

# --- Distance (relative depth -> proximity zones) ---
NEAR_PCTL    = 0.66   # depth >= this percentile -> "near"   (MiDaS: higher = nearer)
FAR_PCTL     = 0.33   # depth <= this percentile -> "far"
CENTER_FRAC  = 0.5    # sample inner 50% of each box for distance

# --- Image handling ---
MAX_IMAGE_SIDE   = 1280  # downscale longest side above this (speed + memory)
DEFAULT_LANGUAGE = "en"

# --- Offline translation (MarianMT, en -> xx). Add more pairs as needed. ---
TRANSLATION_MODELS = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "ru": "Helsinki-NLP/opus-mt-en-ru",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
    "zh": "Helsinki-NLP/opus-mt-en-zh",
}
