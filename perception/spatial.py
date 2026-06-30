"""Spatial reasoning: horizontal position, counts, and a template fallback caption."""
from collections import Counter


def _position(bbox, img_w):
    cx = bbox[0] + bbox[2] / 2
    if cx < img_w / 3:
        return "left"
    if cx > 2 * img_w / 3:
        return "right"
    return "center"


def summarize(objects, img_w):
    """Adds 'position' to each object; returns {counts, object_count}."""
    for o in objects:
        o["position"] = _position(o["bbox"], img_w)
    counts = Counter(o["label"] for o in objects)
    return {"counts": dict(counts), "object_count": len(objects)}


def template_caption(summary, objects):
    """Deterministic fallback used ONLY if the pretrained captioner fails.

    The primary caption comes from BLIP (caption.py); this guarantees the API
    never returns an empty caption even if the model errors out.
    """
    counts = summary["counts"]
    if not counts:
        return "No clear objects detected in the scene."

    def phrase(label, n):
        return f"{n} {label}s" if n > 1 else f"a {label}"

    parts = [phrase(lbl, n) for lbl, n in counts.items()]
    body = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    near = [o["label"] for o in objects if o.get("distance_zone") == "near"]
    tail = f" The {near[0]} is close by." if near else ""
    return f"In the scene there is {body}.{tail}"
