"""Object detection with YOLOv8n (pretrained on COCO, runs locally)."""
from ultralytics import YOLO
import config


class Detector:
    def __init__(self, model_path=config.DETECTOR_MODEL):
        self.model = YOLO(model_path)      # downloads weights once, then cached
        self.names = self.model.names

    def detect(self, image_bgr, min_conf=config.MIN_CONFIDENCE):
        """image_bgr: numpy BGR array. Returns list of detection dicts."""
        results = self.model.predict(
            image_bgr, conf=min_conf, iou=config.NMS_IOU, verbose=False
        )
        out = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append({
                "label": self.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],  # x,y,w,h
            })
        return out


if __name__ == "__main__":
    import sys, cv2
    img = cv2.imread(sys.argv[1])
    for d in Detector().detect(img):
        print(d)
