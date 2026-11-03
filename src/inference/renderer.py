from __future__ import annotations

import cv2
import numpy as np

from src.inference.types import FaceEmotionResult


def _draw_face_results(
    canvas: np.ndarray,
    results: list[FaceEmotionResult],
    *,
    confidence_threshold: float,
) -> None:
    for result in results:
        detection = result.detection
        prediction = result.prediction

        x1 = detection.x
        y1 = detection.y
        x2 = detection.x2
        y2 = detection.y2

        cv2.rectangle(
            canvas,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        prefix = (
            f"#{result.track_id} "
            if result.track_id is not None
            else ""
        )

        if prediction.confidence >= confidence_threshold:
            label = (
                f"{prefix}{prediction.class_name}: "
                f"{prediction.confidence:.2f}"
            )
        else:
            label = (
                f"{prefix}uncertain ({prediction.class_name}: "
                f"{prediction.confidence:.2f})"
            )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        text_y = max(
            text_height + baseline + 4,
            y1,
        )

        cv2.rectangle(
            canvas,
            (x1, max(0, text_y - text_height - baseline - 4)),
            (
                min(canvas.shape[1] - 1, x1 + text_width + 6),
                min(canvas.shape[0] - 1, text_y + 3),
            ),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            canvas,
            label,
            (x1 + 3, text_y - baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def render_frame_results(
    frame_bgr: np.ndarray,
    results: list[FaceEmotionResult],
    *,
    confidence_threshold: float = 0.5,
    processing_fps: float | None = None,
    frame_index: int | None = None,
    stream_fps: float | None = None,
    copy_frame: bool = True,
) -> np.ndarray:
    """Render FER predictions and compact runtime status onto one frame."""

    canvas = frame_bgr.copy() if copy_frame else frame_bgr

    _draw_face_results(
        canvas,
        results,
        confidence_threshold=confidence_threshold,
    )

    if not results:
        cv2.putText(
            canvas,
            "No face detected",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    status_parts: list[str] = []

    if frame_index is not None:
        status_parts.append(f"frame {frame_index}")
    if processing_fps is not None:
        status_parts.append(f"proc {processing_fps:.1f} FPS")
    if stream_fps is not None:
        status_parts.append(f"loop {stream_fps:.1f} FPS")

    if status_parts:
        status = " | ".join(status_parts)
        (text_width, text_height), baseline = cv2.getTextSize(
            status,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1,
        )

        x1 = 8
        y2 = canvas.shape[0] - 8
        y1 = max(0, y2 - text_height - baseline - 8)

        cv2.rectangle(
            canvas,
            (x1, y1),
            (min(canvas.shape[1] - 1, x1 + text_width + 8), y2),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            canvas,
            status,
            (x1 + 4, y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def render_static_results(
    frame_bgr: np.ndarray,
    results: list[FaceEmotionResult],
    *,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    return render_frame_results(
        frame_bgr,
        results,
        confidence_threshold=confidence_threshold,
    )
