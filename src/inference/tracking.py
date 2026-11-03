from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.inference.types import (
    EmotionPrediction,
    FaceDetection,
    FaceEmotionResult,
)


def intersection_over_union(
    first: FaceDetection,
    second: FaceDetection,
) -> float:
    x1 = max(first.x, second.x)
    y1 = max(first.y, second.y)
    x2 = min(first.x2, second.x2)
    y2 = min(first.y2, second.y2)

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection = intersection_width * intersection_height

    union = first.area + second.area - intersection
    if union <= 0:
        return 0.0

    return float(intersection / union)


@dataclass
class _TrackState:
    track_id: int
    detection: FaceDetection
    probabilities: np.ndarray
    missed_frames: int = 0


class TemporalEmotionStabilizer:
    """Lightweight IoU association + EMA smoothing for realtime FER.

    This intentionally avoids heavyweight MOT dependencies. The final lab
    needs stable face-level emotion overlays, not identity re-identification.
    """

    def __init__(
        self,
        *,
        class_names: list[str] | tuple[str, ...],
        iou_threshold: float = 0.30,
        max_missed_frames: int = 10,
        ema_alpha: float = 0.55,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1].")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be >= 0.")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1].")

        self.class_names = tuple(class_names)
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = int(max_missed_frames)
        self.ema_alpha = float(ema_alpha)
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def _new_track(
        self,
        result: FaceEmotionResult,
    ) -> _TrackState:
        probabilities = np.asarray(
            result.prediction.probabilities,
            dtype=np.float32,
        )

        track = _TrackState(
            track_id=self._next_track_id,
            detection=result.detection,
            probabilities=probabilities.copy(),
            missed_frames=0,
        )
        self._tracks[track.track_id] = track
        self._next_track_id += 1
        return track

    def _match(
        self,
        results: list[FaceEmotionResult],
    ) -> dict[int, int]:
        candidates: list[tuple[float, int, int]] = []

        for result_index, result in enumerate(results):
            for track_id, track in self._tracks.items():
                score = intersection_over_union(
                    result.detection,
                    track.detection,
                )
                if score >= self.iou_threshold:
                    candidates.append((score, result_index, track_id))

        candidates.sort(reverse=True)
        matched_results: set[int] = set()
        matched_tracks: set[int] = set()
        matches: dict[int, int] = {}

        for _score, result_index, track_id in candidates:
            if result_index in matched_results or track_id in matched_tracks:
                continue
            matches[result_index] = track_id
            matched_results.add(result_index)
            matched_tracks.add(track_id)

        return matches

    def update(
        self,
        results: tuple[FaceEmotionResult, ...] | list[FaceEmotionResult],
    ) -> tuple[FaceEmotionResult, ...]:
        raw_results = list(results)
        matches = self._match(raw_results)
        matched_track_ids = set(matches.values())

        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_track_ids:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    del self._tracks[track_id]

        stabilized: list[FaceEmotionResult] = []

        for result_index, result in enumerate(raw_results):
            if result_index in matches:
                track = self._tracks[matches[result_index]]
                raw_probabilities = np.asarray(
                    result.prediction.probabilities,
                    dtype=np.float32,
                )
                track.probabilities = (
                    self.ema_alpha * raw_probabilities
                    + (1.0 - self.ema_alpha) * track.probabilities
                )
                probability_sum = float(track.probabilities.sum())
                if probability_sum > 0.0:
                    track.probabilities /= probability_sum
                track.detection = result.detection
                track.missed_frames = 0
            else:
                track = self._new_track(result)

            class_id = int(np.argmax(track.probabilities))
            confidence = float(track.probabilities[class_id])

            prediction = EmotionPrediction(
                class_id=class_id,
                class_name=self.class_names[class_id],
                confidence=confidence,
                probabilities=tuple(
                    float(value)
                    for value in track.probabilities.tolist()
                ),
            )

            stabilized.append(
                FaceEmotionResult(
                    detection=result.detection,
                    prediction=prediction,
                    track_id=track.track_id,
                    raw_prediction=result.prediction,
                )
            )

        return tuple(stabilized)
