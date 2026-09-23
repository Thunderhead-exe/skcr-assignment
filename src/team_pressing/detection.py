"""
Object detection with an Ultralytics YOLO model (YOLO26s by default).
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from team_pressing.config import ModelConfig

# Module logger
log = logging.getLogger(__name__)


@dataclass
class Detections:
    """
    All detections of one frame, as parallel arrays (row i of each array = detection i).

    Attributes:
        xyxy: (N, 4) float32 boxes (x1, y1, x2, y2) in pixels of the frame.
        conf: (N,) float32 confidence scores.
        role: (N,) str role from the class map: person | player | goalkeeper | referee | ball.
    """

    xyxy: np.ndarray
    conf: np.ndarray
    role: np.ndarray

    def __len__(self) -> int:
        """Number of detections."""
        return len(self.conf)

    def select(self, mask: np.ndarray) -> "Detections":
        """Subset of the detections (boolean mask or index array)."""
        return Detections(self.xyxy[mask], self.conf[mask], self.role[mask])

    @staticmethod
    def empty() -> "Detections":
        """No detections (correctly shaped empty arrays)."""
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, dtype="<U10"))


def resolve_device(device: str) -> str:
    """
    Turn "auto" into the best available device: CUDA GPU, then Apple GPU (MPS), then CPU.

    Any other value (e.g. "cpu", "mps", "cuda:0") is returned unchanged.
    """
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file, read in 1 MB chunks (no need to load it all in memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Detector:
    """Loads the YOLO model once and turns RGB frames into Detections."""

    def __init__(self, cfg: ModelConfig):
        """
        Load the weights, resolve the device and precompute what every predict call needs.

        Args:
            cfg: model section of the configuration.
        """
        self.cfg = cfg
        self.device = resolve_device(cfg.device)  # actual device used for inference
        Path(cfg.weights).parent.mkdir(parents=True, exist_ok=True)  # e.g. models/
        # Official weight names (yolo26s.pt, ...) are downloaded from the Ultralytics release on first use.
        self.model = YOLO(cfg.weights)
        self.class_ids = sorted(cfg.class_map)  # model class ids to keep (e.g. [0, 32] for COCO)
        # The model call uses the lowest per-role threshold; stricter ones are applied afterwards.
        self.min_conf = min(cfg.conf[cfg.class_map[c]] for c in self.class_ids)
        self.weights_sha256 = file_sha256(cfg.weights)  # identifies the exact weights used
        names = self.model.names  # model class id -> class name (e.g. 32 -> "sports ball")
        log.info(
            "Model: %s (sha256 %s...) task=%s device=%s imgsz=%d classes=%s",
            cfg.weights,
            self.weights_sha256[:16],
            self.model.task,
            self.device,
            cfg.imgsz,
            {c: f"{names.get(c, '?')}->{cfg.class_map[c]}" for c in self.class_ids},
        )

    def info(self) -> dict:
        """Model description for metrics.json."""
        return {
            "weights": self.cfg.weights,
            "sha256": self.weights_sha256,
            "device": self.device,
            "imgsz": self.cfg.imgsz,
            "class_map": self.cfg.class_map,
        }

    def detect(self, frame_rgb: np.ndarray) -> Detections:
        """
        Detect the configured classes in one RGB frame.

        Args:
            frame_rgb: 720x1280 RGB uint8 frame.
        """
        # Step 1: inference.
        # Ultralytics treats numpy arrays as OpenCV BGR and flips them to RGB internally.
        # Our frames are RGB, so hand over a BGR view; passing RGB directly would silently
        # feed the network swapped channels and degrade accuracy.
        result = self.model.predict(
            np.ascontiguousarray(frame_rgb[..., ::-1]),
            imgsz=self.cfg.imgsz,  # network input size (1280 = no downscaling of the frame width)
            conf=self.min_conf,  # lowest threshold; per-role filtering below
            classes=self.class_ids,  # restrict the head output to the classes we use
            device=self.device,
            verbose=False,  # no per-frame console output from Ultralytics
        )[0]  # one image in, one result out
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty()

        # Step 2: move the results from the GPU to numpy and map class ids to pipeline roles.
        cls = boxes.cls.cpu().numpy().astype(int)  # model class id per box
        dets = Detections(
            xyxy=boxes.xyxy.cpu().numpy().astype(np.float32),
            conf=boxes.conf.cpu().numpy().astype(np.float32),
            role=np.array([self.cfg.class_map[c] for c in cls], dtype="<U10"),
        )

        # Step 3: per-role confidence thresholds (e.g. 0.30 for people, 0.15 for the ball).
        thresholds = np.array([self.cfg.conf[r] for r in dets.role], dtype=np.float32)
        dets = dets.select(dets.conf >= thresholds)

        # Step 4: there is a single ball on the pitch: keep only the most confident candidate.
        ball_idx = np.flatnonzero(dets.role == "ball")
        if len(ball_idx) > 1:
            keep = np.ones(len(dets), bool)  # keep everything...
            keep[ball_idx] = False  # ...except the balls...
            keep[ball_idx[np.argmax(dets.conf[ball_idx])]] = True  # ...but the best one
            dets = dets.select(keep)
        return dets
