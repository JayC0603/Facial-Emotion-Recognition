from __future__ import annotations

import cv2
import numpy as np
import torch

from src.inference.types import FaceDetection


def square_face_crop(
    frame_bgr: np.ndarray,
    detection: FaceDetection,
    *,
    padding_ratio: float = 0.10,
) -> np.ndarray:
    """Crop a padded square face region while staying inside the frame."""

    if padding_ratio < 0.0:
        raise ValueError("padding_ratio must be >= 0.")

    frame_height, frame_width = frame_bgr.shape[:2]
    center_x = detection.x + detection.width / 2.0
    center_y = detection.y + detection.height / 2.0
    side = max(detection.width, detection.height) * (
        1.0 + 2.0 * padding_ratio
    )
    side = max(2.0, side)

    x1 = max(0, int(np.floor(center_x - side / 2.0)))
    y1 = max(0, int(np.floor(center_y - side / 2.0)))
    x2 = min(frame_width, int(np.ceil(center_x + side / 2.0)))
    y2 = min(frame_height, int(np.ceil(center_y + side / 2.0)))

    if x2 - x1 <= 1 or y2 - y1 <= 1:
        raise RuntimeError(
            "Face crop is empty after clamping to the frame."
        )

    # The slice is usually non-contiguous after later channel reordering, so
    # make the crop contiguous once here. This also decouples it from a frame
    # that the renderer may mutate in-place after inference.
    return np.ascontiguousarray(frame_bgr[y1:y2, x1:x2])


class InferencePreprocessor:
    """Deterministic and allocation-conscious inference preprocessing.

    Contract:
        OpenCV BGR face crop
        -> BGR-to-grayscale
        -> resize to input_size x input_size
        -> float32 / 255
        -> normalize with mean/std 0.5
        -> contiguous NCHW tensor with one channel

    The batch path preallocates one NumPy buffer and normalizes all faces
    together. This removes per-face Torch tensors + torch.stack allocations
    from the old implementation while preserving the validation contract.
    """

    def __init__(self, input_size: int = 64) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        self.input_size = int(input_size)

    def _resize_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        if (height, width) == (self.input_size, self.input_size):
            return image_bgr

        downscale = height > self.input_size or width > self.input_size
        interpolation = cv2.INTER_AREA if downscale else cv2.INTER_CUBIC
        return cv2.resize(
            image_bgr,
            (self.input_size, self.input_size),
            interpolation=interpolation,
        )

    def prepare_face(self, face_bgr: np.ndarray) -> torch.Tensor:
        return self.prepare_batch([face_bgr])[0]

    def prepare_batch(
        self,
        faces_bgr: list[np.ndarray],
    ) -> torch.Tensor:
        count = len(faces_bgr)
        if count == 0:
            return torch.empty(
                (0, 1, self.input_size, self.input_size),
                dtype=torch.float32,
            )

        batch_gray = np.empty(
            (count, self.input_size, self.input_size),
            dtype=np.float32,
        )

        for index, face_bgr in enumerate(faces_bgr):
            if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
                raise ValueError(
                    "Expected BGR face crop with shape HxWx3."
                )
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            batch_gray[index] = self._resize_bgr(gray)

        batch_gray *= np.float32(1.0 / 255.0)
        batch_gray -= np.float32(0.5)
        batch_gray /= np.float32(0.5)

        nchw = np.ascontiguousarray(
            batch_gray[:, None, :, :]
        )
        return torch.from_numpy(nchw)
