"""Monocular depth with MiDaS (pretrained, runs locally).

NOTE: MiDaS outputs RELATIVE inverse-depth (disparity-like), NOT metric meters.
Higher value = nearer. We turn this into near/medium/far zones in fusion.py.
"""
import cv2
import torch
import config


class DepthEstimator:
    def __init__(self, model_name=config.DEPTH_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load("intel-isl/MiDaS", model_name)
        self.model.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = (
            transforms.small_transform if "small" in model_name
            else transforms.dpt_transform
        )

    def estimate(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=rgb.shape[:2],
                mode="bicubic", align_corners=False,
            ).squeeze()
        return pred.cpu().numpy()      # relative inverse-depth (higher = nearer)


if __name__ == "__main__":
    import sys, cv2
    img = cv2.imread(sys.argv[1])
    d = DepthEstimator().estimate(img)
    vis = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    cv2.imwrite("depth_vis.png", cv2.applyColorMap(vis, cv2.COLORMAP_MAGMA))
    print("saved depth_vis.png", d.shape)
