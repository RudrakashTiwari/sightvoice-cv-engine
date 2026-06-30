"""Image ingestion: decode bytes, fix EXIF orientation, optionally downscale."""
import io
import cv2
import numpy as np
from PIL import Image, ImageOps
import config


def decode(image_bytes):
    """Return (bgr_ndarray, pil_rgb_image, (width, height)).

    - Fixes EXIF rotation (phone photos are often silently rotated).
    - Downscales the longest side to MAX_IMAGE_SIDE so detection, depth and
      caption all operate on the SAME image -> coordinates stay consistent.
    """
    pil = Image.open(io.BytesIO(image_bytes))
    pil = ImageOps.exif_transpose(pil).convert("RGB")

    w, h = pil.size
    longest = max(w, h)
    if longest > config.MAX_IMAGE_SIDE:
        scale = config.MAX_IMAGE_SIDE / longest
        w, h = int(w * scale), int(h * scale)
        pil = pil.resize((w, h), Image.BILINEAR)

    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, pil, (w, h)
