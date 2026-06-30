"""Fuse detections + depth map -> per-object proximity zone."""
import numpy as np
import config


def _center_region(bbox, frac=config.CENTER_FRAC):
    """Inner region of a box, to avoid sampling background pixels at the edges."""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    hw, hh = w * frac / 2, h * frac / 2
    return int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)


def add_distances(objects, depth_map):
    """Annotate each object with distance_zone (near/medium/far) and distance_m.

    Graceful degradation: if depth_map is None, distances are null but objects
    are still returned (spec FR-8).
    """
    if depth_map is None:
        for o in objects:
            o["distance_zone"], o["distance_m"] = None, None
        return objects

    near_t = float(np.quantile(depth_map, config.NEAR_PCTL))
    far_t  = float(np.quantile(depth_map, config.FAR_PCTL))

    for o in objects:
        x1, y1, x2, y2 = _center_region(o["bbox"])
        patch = depth_map[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
        val = float(np.median(patch)) if patch.size else float(np.median(depth_map))
        # MiDaS: higher value = nearer
        if val >= near_t:
            zone = "near"
        elif val <= far_t:
            zone = "far"
        else:
            zone = "medium"
        o["distance_zone"] = zone
        o["distance_m"] = None      # relative depth only; metric is a future upgrade
    return objects
