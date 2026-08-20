import time
from ultralytics import YOLO

# COCO class ids we care about
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}


class VehicleDetector:
    def __init__(self, model_path="yolo11n.pt", confidence=0.4, device=None):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device or self._pick_device()

    def _pick_device(self):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def set_confidence(self, confidence):
        self.confidence = confidence

    def track(self, frame):
        """
        Run detection + ByteTrack tracking on a single frame.
        Returns (results_list, inference_ms).
        """
        start = time.time()
        results = self.model.track(
            frame,
            persist=True,
            classes=list(VEHICLE_CLASSES.keys()),
            conf=self.confidence,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )
        inference_ms = (time.time() - start) * 1000
        return self._parse(results), inference_ms

    def _parse(self, results):
        """Extract (track_id, cls_name, conf, x1, y1, x2, y2, cx, cy) tuples."""
        detections = []
        if not results:
            return detections
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return detections

        ids = r.boxes.id.int().tolist()
        classes = r.boxes.cls.int().tolist()
        confs = r.boxes.conf.tolist()
        xyxy = r.boxes.xyxy.tolist()

        for tid, cls, conf, box in zip(ids, classes, confs, xyxy):
            if cls not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append({
                "track_id": tid,
                "class_name": VEHICLE_CLASSES[cls],
                "confidence": conf,
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "center": (cx, cy),
            })
        return detections
