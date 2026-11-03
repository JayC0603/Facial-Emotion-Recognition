from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """One face detection in source-frame pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    score: float

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x2, self.y2


@dataclass(frozen=True)
class EmotionPrediction:
    """Classifier output for one face crop."""

    class_id: int
    class_name: str
    confidence: float
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class FaceEmotionResult:
    """Detection + classifier result for one face.

    ``track_id`` and ``raw_prediction`` are optional so the exact same type is
    usable for P8.1/P8.2 raw inference and P8.4 temporally stabilised output.
    """

    detection: FaceDetection
    prediction: EmotionPrediction
    track_id: int | None = None
    raw_prediction: EmotionPrediction | None = None

    def to_dict(self) -> dict:
        payload = {
            "bbox": {
                "x": self.detection.x,
                "y": self.detection.y,
                "width": self.detection.width,
                "height": self.detection.height,
            },
            "face_score": self.detection.score,
            "emotion": self.prediction.class_name,
            "class_id": self.prediction.class_id,
            "confidence": self.prediction.confidence,
            "probabilities": list(self.prediction.probabilities),
        }

        if self.track_id is not None:
            payload["track_id"] = int(self.track_id)

        if self.raw_prediction is not None:
            payload["raw_prediction"] = {
                "emotion": self.raw_prediction.class_name,
                "class_id": self.raw_prediction.class_id,
                "confidence": self.raw_prediction.confidence,
                "probabilities": list(self.raw_prediction.probabilities),
            }

        return payload
