from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.inference.types import FaceDetection


class YuNetFaceDetector:
    """OpenCV YuNet wrapper with optional detector-frame downscaling.

    Emotion crops always come from the original frame. Only YuNet's detector
    input may be downscaled, which is a major CPU optimization for 1080p/4K
    camera frames. Returned boxes are mapped back to original coordinates.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        score_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        max_input_side: int | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                "YuNet model not found:\n"
                f"{self.model_path}"
            )
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1].")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be in [0, 1].")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if max_input_side is not None and max_input_side <= 0:
            raise ValueError("max_input_side must be positive when provided.")

        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.max_input_side = (
            int(max_input_side) if max_input_side is not None else None
        )

        creator = getattr(cv2, "FaceDetectorYN_create", None)
        if creator is None:
            face_detector_yn = getattr(cv2, "FaceDetectorYN", None)
            creator = getattr(face_detector_yn, "create", None)
        if creator is None:
            raise RuntimeError(
                "This OpenCV build does not expose FaceDetectorYN. "
                "Install/update opencv-python."
            )

        self._detector = creator(
            str(self.model_path),
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
        )
        self._input_size: tuple[int, int] | None = None

    @staticmethod
    def _clamp_detection(
        row: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> FaceDetection | None:
        x, y, width, height = map(float, row[:4])
        score = float(row[-1])

        x1 = max(0, min(frame_width, int(np.floor(x))))
        y1 = max(0, min(frame_height, int(np.floor(y))))
        x2 = max(0, min(frame_width, int(np.ceil(x + width))))
        y2 = max(0, min(frame_height, int(np.ceil(y + height))))
        clipped_width = x2 - x1
        clipped_height = y2 - y1
        if clipped_width <= 1 or clipped_height <= 1:
            return None

        return FaceDetection(
            x=x1,
            y=y1,
            width=clipped_width,
            height=clipped_height,
            score=score,
        )

    def _detector_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        frame_height, frame_width = frame_bgr.shape[:2]
        if (
            self.max_input_side is None
            or max(frame_height, frame_width) <= self.max_input_side
        ):
            return frame_bgr, 1.0, 1.0

        scale = self.max_input_side / float(max(frame_height, frame_width))
        detect_width = max(2, int(round(frame_width * scale)))
        detect_height = max(2, int(round(frame_height * scale)))
        resized = cv2.resize(
            frame_bgr,
            (detect_width, detect_height),
            interpolation=cv2.INTER_AREA,
        )
        return (
            resized,
            frame_width / float(detect_width),
            frame_height / float(detect_height),
        )

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a numpy array.")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("YuNet expects a BGR image with shape HxWx3.")

        frame_height, frame_width = frame_bgr.shape[:2]
        if frame_height <= 1 or frame_width <= 1:
            return []

        detector_frame, scale_x, scale_y = self._detector_frame(frame_bgr)
        detect_height, detect_width = detector_frame.shape[:2]
        input_size = (int(detect_width), int(detect_height))
        if input_size != self._input_size:
            self._detector.setInputSize(input_size)
            self._input_size = input_size

        _, faces = self._detector.detect(detector_frame)
        if faces is None or len(faces) == 0:
            return []

        detections: list[FaceDetection] = []
        for row in np.asarray(faces):
            if scale_x != 1.0 or scale_y != 1.0:
                mapped = np.asarray(row, dtype=np.float32).copy()
                mapped[0] *= scale_x
                mapped[2] *= scale_x
                mapped[1] *= scale_y
                mapped[3] *= scale_y
                row = mapped

            detection = self._clamp_detection(
                row,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if detection is not None:
                detections.append(detection)

        detections.sort(
            key=lambda item: (item.score, item.area),
            reverse=True,
        )
        return detections
