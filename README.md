# Vision Assistance — CV Perception Engine

Offline computer-vision backend for an assistive-vision app. Takes a single image and
returns detected objects, per-object proximity, and a natural-language scene caption
(translatable). **Everything runs on pretrained models locally — no API keys, no external
services at inference time.**

## What it does
- **Object detection** — YOLOv8n (COCO, 80 classes)
- **Distance** — MiDaS depth → `near` / `medium` / `far` zones per object
- **Scene caption** — BLIP image-captioning model (pretrained, runs on the image)
- **Multilingual** — MarianMT offline translation (en → hi/fr/de/es/ru/ar/zh)
- **API** — Flask `POST /analyze`

> First run downloads model weights from the Ultralytics / PyTorch Hub / Hugging Face
> caches (one-time internet). After that, inference is fully offline. No API key is ever used.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the API
```bash
python app.py        # serves on http://localhost:5000
```
Test it:
```bash
curl -F "image=@tests/fixtures/desk.jpg" http://localhost:5000/analyze
curl -F "image=@tests/fixtures/desk.jpg" -F "language=hi" http://localhost:5000/analyze
```

## Run on one image (no server)
```bash
python tests/run_local.py tests/fixtures/desk.jpg hi
```

## Test individual modules
```bash
python -m perception.detector tests/fixtures/desk.jpg   # prints detections
python -m perception.depth    tests/fixtures/desk.jpg   # writes depth_vis.png
python -m perception.caption  tests/fixtures/desk.jpg   # prints BLIP caption
```

## API contract
`POST /analyze` (multipart): `image` (file, required), `language` (optional, default `en`),
`min_confidence` (optional float).

Response:
```json
{
  "status": "ok",
  "image": { "width": 1280, "height": 720 },
  "objects": [
    { "label": "laptop", "confidence": 0.91, "bbox": [430,150,180,120],
      "position": "center", "distance_zone": "near", "distance_m": null }
  ],
  "summary": { "counts": { "laptop": 1 }, "object_count": 1 },
  "caption": { "language": "hi", "text": "...", "text_en": "a laptop on a desk" },
  "meta": { "model_detector": "yolov8n.pt", "model_depth": "MiDaS_small",
            "model_caption": "Salesforce/blip-image-captioning-base", "timing_ms": 720 }
}
```

## Project structure
```
vision-cv-engine/
├── app.py                 # Flask /analyze + /health
├── config.py              # models, thresholds, language map
├── requirements.txt
├── perception/
│   ├── preprocess.py      # decode, EXIF fix, downscale
│   ├── detector.py        # YOLOv8n
│   ├── depth.py           # MiDaS
│   ├── fusion.py          # depth -> near/medium/far
│   ├── spatial.py         # position, counts, template fallback
│   ├── caption.py         # BLIP scene caption
│   ├── translate.py       # MarianMT offline translation
│   └── pipeline.py        # orchestration + JSON assembly
└── tests/
    ├── run_local.py       # run pipeline on one image
    └── fixtures/          # put your test photos here
```

## Notes
- MiDaS gives **relative** depth, not meters — that is why distance is reported as
  near/medium/far. `distance_m` is reserved for a future metric-calibration upgrade.
- The caption comes from the pretrained BLIP model; a deterministic template caption is used
  only as a fallback if the model errors, so the API never returns an empty caption.
