"""Orchestrates the full pipeline and assembles the spec JSON response."""
import time
import concurrent.futures as cf
import config
from .preprocess import decode
from .detector import Detector
from .depth import DepthEstimator
from .fusion import add_distances
from .spatial import summarize, template_caption
from .caption import Captioner
from .translate import Translator


class Pipeline:
    def __init__(self):
        # Load every model ONCE at startup, never per request.
        self.detector = Detector()
        self.depth = DepthEstimator()
        self.captioner = Captioner()
        self.translator = Translator()

    def run(self, image_bytes, language=config.DEFAULT_LANGUAGE, min_conf=None):
        t0 = time.perf_counter()
        bgr, pil, (w, h) = decode(image_bytes)

        # Detection and depth are independent -> run them concurrently.
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            f_det = ex.submit(self.detector.detect, bgr, min_conf or config.MIN_CONFIDENCE)
            f_dep = ex.submit(self._safe_depth, bgr)
            objects = f_det.result()
            depth_map = f_dep.result()

        objects = add_distances(objects, depth_map)
        summary = summarize(objects, w)

        # Primary caption = pretrained BLIP. Fallback = template (never empty).
        try:
            caption_en = self.captioner.caption(pil)
        except Exception:
            caption_en = template_caption(summary, objects)

        caption_text = self.translator.translate(caption_en, language)

        return {
            "status": "ok",
            "image": {"width": w, "height": h},
            "objects": objects,
            "summary": summary,
            "caption": {
                "language": language,
                "text": caption_text,
                "text_en": caption_en,
            },
            "meta": {
                "model_detector": config.DETECTOR_MODEL,
                "model_depth": config.DEPTH_MODEL,
                "model_caption": config.CAPTION_MODEL,
                "timing_ms": int((time.perf_counter() - t0) * 1000),
            },
        }

    def _safe_depth(self, bgr):
        try:
            return self.depth.estimate(bgr)
        except Exception:
            return None      # graceful degradation (spec FR-8)
