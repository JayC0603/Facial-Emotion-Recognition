from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import cv2
from tqdm.auto import tqdm

from src.inference.detector import YuNetFaceDetector
from src.inference.predictor import EmotionPredictor
from src.inference.preprocess import (
    InferencePreprocessor,
    square_face_crop,
)
from src.inference.renderer import render_frame_results
from src.inference.sources import (
    LocalVideoReader,
    NetworkStreamReader,
)
from src.inference.tracking import TemporalEmotionStabilizer
from src.inference.types import FaceEmotionResult


@dataclass(frozen=True)
class FrameInferenceOutput:
    frame_index: int
    num_faces: int
    processing_seconds: float
    results: tuple[FaceEmotionResult, ...]
    detector_seconds: float = 0.0
    preprocess_seconds: float = 0.0
    classifier_seconds: float = 0.0

    @property
    def processing_fps(self) -> float:
        return 1.0 / max(self.processing_seconds, 1e-9)

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "num_faces": self.num_faces,
            "processing_seconds": self.processing_seconds,
            "processing_fps": self.processing_fps,
            "timings": {
                "detector_seconds": self.detector_seconds,
                "preprocess_seconds": self.preprocess_seconds,
                "classifier_seconds": self.classifier_seconds,
            },
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class StaticImageInferenceOutput:
    source_path: str
    output_path: str
    width: int
    height: int
    num_faces: int
    results: tuple[FaceEmotionResult, ...]

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "output_path": self.output_path,
            "image_size": {
                "width": self.width,
                "height": self.height,
            },
            "num_faces": self.num_faces,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class VideoInferenceOutput:
    source_path: str
    output_path: str
    frames_jsonl_path: str
    width: int
    height: int
    source_fps: float
    frames_processed: int
    total_faces: int
    wall_seconds: float
    processing_fps: float
    average_faces_per_frame: float
    stopped_early: bool
    temporal_stabilization: bool

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "output_path": self.output_path,
            "frames_jsonl_path": self.frames_jsonl_path,
            "video": {
                "width": self.width,
                "height": self.height,
                "source_fps": self.source_fps,
                "frames_processed": self.frames_processed,
            },
            "runtime": {
                "wall_seconds": self.wall_seconds,
                "processing_fps": self.processing_fps,
                "total_faces": self.total_faces,
                "average_faces_per_frame": self.average_faces_per_frame,
                "stopped_early": self.stopped_early,
                "temporal_stabilization": self.temporal_stabilization,
            },
        }


@dataclass(frozen=True)
class StreamInferenceOutput:
    source: str
    output_path: str | None
    frames_jsonl_path: str | None
    frames_processed: int
    total_faces: int
    wall_seconds: float
    processing_fps: float
    average_faces_per_frame: float
    reconnect_count: int
    frames_captured: int
    stopped_early: bool
    temporal_stabilization: bool

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "output_path": self.output_path,
            "frames_jsonl_path": self.frames_jsonl_path,
            "runtime": {
                "frames_processed": self.frames_processed,
                "wall_seconds": self.wall_seconds,
                "processing_fps": self.processing_fps,
                "total_faces": self.total_faces,
                "average_faces_per_frame": self.average_faces_per_frame,
                "reconnect_count": self.reconnect_count,
                "frames_captured": self.frames_captured,
                "stopped_early": self.stopped_early,
                "temporal_stabilization": self.temporal_stabilization,
            },
        }


class FrameEmotionProcessor:
    """Reusable detector -> crop -> batched emotion processor for one frame."""

    def __init__(
        self,
        *,
        detector: YuNetFaceDetector,
        predictor: EmotionPredictor,
        crop_padding_ratio: float = 0.10,
    ) -> None:
        if crop_padding_ratio < 0.0:
            raise ValueError("crop_padding_ratio must be >= 0.")

        self.detector = detector
        self.predictor = predictor
        self.preprocessor = InferencePreprocessor(
            input_size=predictor.input_size
        )
        self.crop_padding_ratio = float(crop_padding_ratio)

    def process(
        self,
        frame_bgr,
        *,
        frame_index: int = 0,
    ) -> FrameInferenceOutput:
        start_time = perf_counter()

        detector_start = perf_counter()
        detections = self.detector.detect(frame_bgr)
        detector_seconds = perf_counter() - detector_start

        preprocess_start = perf_counter()
        valid_detections = []
        face_crops = []

        for detection in detections:
            try:
                crop = square_face_crop(
                    frame_bgr,
                    detection,
                    padding_ratio=self.crop_padding_ratio,
                )
            except RuntimeError:
                continue

            valid_detections.append(detection)
            face_crops.append(crop)

        batch = self.preprocessor.prepare_batch(face_crops)
        preprocess_seconds = perf_counter() - preprocess_start

        classifier_start = perf_counter()
        predictions = self.predictor.predict_batch(batch)
        classifier_seconds = perf_counter() - classifier_start

        if len(predictions) != len(valid_detections):
            raise RuntimeError("Detector/predictor batch length mismatch.")

        results = tuple(
            FaceEmotionResult(
                detection=detection,
                prediction=prediction,
            )
            for detection, prediction in zip(
                valid_detections,
                predictions,
            )
        )

        elapsed = perf_counter() - start_time
        return FrameInferenceOutput(
            frame_index=int(frame_index),
            num_faces=len(results),
            processing_seconds=float(elapsed),
            results=results,
            detector_seconds=float(detector_seconds),
            preprocess_seconds=float(preprocess_seconds),
            classifier_seconds=float(classifier_seconds),
        )


class StaticImageEmotionPipeline:
    def __init__(
        self,
        *,
        detector: YuNetFaceDetector,
        predictor: EmotionPredictor,
        crop_padding_ratio: float = 0.10,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.processor = FrameEmotionProcessor(
            detector=detector,
            predictor=predictor,
            crop_padding_ratio=crop_padding_ratio,
        )
        self.confidence_threshold = float(confidence_threshold)

    def run(
        self,
        *,
        source_path: Path,
        output_path: Path,
    ) -> StaticImageInferenceOutput:
        source_path = Path(source_path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()

        if not source_path.is_file():
            raise FileNotFoundError(f"Input image not found: {source_path}")

        frame_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"OpenCV failed to decode: {source_path}")

        frame_output = self.processor.process(frame_bgr, frame_index=0)
        rendered = render_frame_results(
            frame_bgr,
            list(frame_output.results),
            confidence_threshold=self.confidence_threshold,
            copy_frame=False,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), rendered):
            raise RuntimeError(f"Failed to save rendered image: {output_path}")

        height, width = frame_bgr.shape[:2]
        return StaticImageInferenceOutput(
            source_path=str(source_path),
            output_path=str(output_path),
            width=int(width),
            height=int(height),
            num_faces=frame_output.num_faces,
            results=frame_output.results,
        )


class VideoEmotionPipeline:
    """Local-video pipeline with optional P8.4 temporal stabilization."""

    def __init__(
        self,
        *,
        detector: YuNetFaceDetector,
        predictor: EmotionPredictor,
        crop_padding_ratio: float = 0.10,
        confidence_threshold: float = 0.5,
        output_codec: str = "mp4v",
        stabilizer: TemporalEmotionStabilizer | None = None,
    ) -> None:
        if len(output_codec) != 4:
            raise ValueError("output_codec must contain exactly four characters.")

        self.processor = FrameEmotionProcessor(
            detector=detector,
            predictor=predictor,
            crop_padding_ratio=crop_padding_ratio,
        )
        self.confidence_threshold = float(confidence_threshold)
        self.output_codec = str(output_codec)
        self.stabilizer = stabilizer

    def run(
        self,
        *,
        source_path: Path,
        output_path: Path,
        max_frames: int | None = None,
        display: bool = False,
    ) -> VideoInferenceOutput:
        source_path = Path(source_path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()

        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames_jsonl_path = output_path.with_suffix(".jsonl")
        if self.stabilizer is not None:
            self.stabilizer.reset()

        frames_processed = 0
        total_faces = 0
        stopped_early = False
        wall_start = perf_counter()

        with LocalVideoReader(source_path) as reader:
            metadata = reader.metadata
            writer = _open_video_writer(
                output_path,
                codec=self.output_codec,
                fps=metadata.fps,
                width=metadata.width,
                height=metadata.height,
            )

            total = metadata.frame_count
            if max_frames is not None:
                total = min(total, max_frames) if total is not None else max_frames

            progress = tqdm(total=total, desc="P8 video", unit="frame")
            try:
                with frames_jsonl_path.open("w", encoding="utf-8") as jsonl:
                    for frame_index, frame_bgr in reader.frames():
                        if max_frames is not None and frames_processed >= max_frames:
                            stopped_early = True
                            break

                        frame_output = self.processor.process(
                            frame_bgr,
                            frame_index=frame_index,
                        )
                        results = frame_output.results
                        if self.stabilizer is not None:
                            results = self.stabilizer.update(results)

                        rendered = render_frame_results(
                            frame_bgr,
                            list(results),
                            confidence_threshold=self.confidence_threshold,
                            processing_fps=frame_output.processing_fps,
                            frame_index=frame_index,
                            copy_frame=False,
                        )
                        writer.write(rendered)

                        payload = frame_output.to_dict()
                        payload["results"] = [result.to_dict() for result in results]
                        payload["temporal_stabilization"] = self.stabilizer is not None
                        jsonl.write(json.dumps(payload) + "\n")

                        total_faces += len(results)
                        frames_processed += 1
                        progress.update(1)

                        if display:
                            cv2.imshow("FER", rendered)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                stopped_early = True
                                break
            finally:
                progress.close()
                writer.release()
                if display:
                    cv2.destroyAllWindows()

        wall_seconds = perf_counter() - wall_start
        processing_fps = frames_processed / max(wall_seconds, 1e-9)
        average_faces = total_faces / max(frames_processed, 1)

        return VideoInferenceOutput(
            source_path=str(source_path),
            output_path=str(output_path),
            frames_jsonl_path=str(frames_jsonl_path),
            width=metadata.width,
            height=metadata.height,
            source_fps=metadata.fps,
            frames_processed=frames_processed,
            total_faces=total_faces,
            wall_seconds=float(wall_seconds),
            processing_fps=float(processing_fps),
            average_faces_per_frame=float(average_faces),
            stopped_early=bool(stopped_early),
            temporal_stabilization=self.stabilizer is not None,
        )


class StreamEmotionPipeline:
    """P8.3 IP/RTSP latest-frame pipeline with reconnect and P8.4 EMA."""

    def __init__(
        self,
        *,
        detector: YuNetFaceDetector,
        predictor: EmotionPredictor,
        crop_padding_ratio: float = 0.10,
        confidence_threshold: float = 0.5,
        output_codec: str = "mp4v",
        stabilizer: TemporalEmotionStabilizer | None = None,
        reconnect_initial_seconds: float = 0.5,
        reconnect_max_seconds: float = 5.0,
        stream_timeout_seconds: float = 10.0,
        record_fps: float = 30.0,
        stream_width: int | None = None,
        stream_height: int | None = None,
    ) -> None:
        self.processor = FrameEmotionProcessor(
            detector=detector,
            predictor=predictor,
            crop_padding_ratio=crop_padding_ratio,
        )
        self.confidence_threshold = float(confidence_threshold)
        self.output_codec = str(output_codec)
        self.stabilizer = stabilizer
        self.reconnect_initial_seconds = float(reconnect_initial_seconds)
        self.reconnect_max_seconds = float(reconnect_max_seconds)
        self.stream_timeout_seconds = float(stream_timeout_seconds)
        self.record_fps = float(record_fps)
        self.stream_width = int(stream_width) if stream_width is not None else None
        self.stream_height = int(stream_height) if stream_height is not None else None
        if self.stream_width is not None and self.stream_width <= 0:
            raise ValueError("stream_width must be positive when provided.")
        if self.stream_height is not None and self.stream_height <= 0:
            raise ValueError("stream_height must be positive when provided.")

    def run(
        self,
        *,
        source: str,
        output_path: Path | None = None,
        max_frames: int | None = None,
        display: bool = True,
    ) -> StreamInferenceOutput:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided.")
        if not display and output_path is None:
            raise ValueError(
                "A stream run must either display frames or record --output."
            )

        if self.stabilizer is not None:
            self.stabilizer.reset()

        output_path = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else None
        )
        frames_jsonl_path = (
            output_path.with_suffix(".jsonl")
            if output_path is not None
            else None
        )
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = None
        jsonl = None
        frames_processed = 0
        total_faces = 0
        stopped_early = False
        wall_start = perf_counter()

        with NetworkStreamReader(
            source,
            reconnect_initial_seconds=self.reconnect_initial_seconds,
            reconnect_max_seconds=self.reconnect_max_seconds,
            target_width=self.stream_width,
            target_height=self.stream_height,
        ) as reader:
            try:
                if frames_jsonl_path is not None:
                    jsonl = frames_jsonl_path.open("w", encoding="utf-8")

                for sequence, frame_bgr in reader.frames(
                    timeout_seconds=self.stream_timeout_seconds
                ):
                    if max_frames is not None and frames_processed >= max_frames:
                        stopped_early = True
                        break

                    frame_output = self.processor.process(
                        frame_bgr,
                        frame_index=sequence,
                    )
                    results = frame_output.results
                    if self.stabilizer is not None:
                        results = self.stabilizer.update(results)

                    loop_elapsed = perf_counter() - wall_start
                    loop_fps = (frames_processed + 1) / max(loop_elapsed, 1e-9)

                    rendered = render_frame_results(
                        frame_bgr,
                        list(results),
                        confidence_threshold=self.confidence_threshold,
                        processing_fps=frame_output.processing_fps,
                        frame_index=sequence,
                        stream_fps=loop_fps,
                        copy_frame=False,
                    )

                    if output_path is not None:
                        if writer is None:
                            height, width = rendered.shape[:2]
                            writer = _open_video_writer(
                                output_path,
                                codec=self.output_codec,
                                fps=self.record_fps,
                                width=width,
                                height=height,
                            )
                        writer.write(rendered)

                    if jsonl is not None:
                        payload = frame_output.to_dict()
                        payload["results"] = [result.to_dict() for result in results]
                        payload["stream_sequence"] = int(sequence)
                        payload["temporal_stabilization"] = self.stabilizer is not None
                        jsonl.write(json.dumps(payload) + "\n")

                    total_faces += len(results)
                    frames_processed += 1

                    if display:
                        cv2.imshow("FER realtime", rendered)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stopped_early = True
                            break
            finally:
                statistics = reader.statistics()
                if writer is not None:
                    writer.release()
                if jsonl is not None:
                    jsonl.close()
                if display:
                    cv2.destroyAllWindows()

        wall_seconds = perf_counter() - wall_start
        processing_fps = frames_processed / max(wall_seconds, 1e-9)
        average_faces = total_faces / max(frames_processed, 1)

        return StreamInferenceOutput(
            source=source,
            output_path=str(output_path) if output_path is not None else None,
            frames_jsonl_path=(
                str(frames_jsonl_path)
                if frames_jsonl_path is not None
                else None
            ),
            frames_processed=frames_processed,
            total_faces=total_faces,
            wall_seconds=float(wall_seconds),
            processing_fps=float(processing_fps),
            average_faces_per_frame=float(average_faces),
            reconnect_count=statistics.reconnect_count,
            frames_captured=statistics.frames_captured,
            stopped_early=bool(stopped_early),
            temporal_stabilization=self.stabilizer is not None,
        )


def _open_video_writer(
    output_path: Path,
    *,
    codec: str,
    fps: float,
    width: int,
    height: int,
):
    if len(codec) != 4:
        raise ValueError("Video codec must contain exactly four characters.")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        float(max(fps, 1.0)),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"OpenCV failed to open video writer: {output_path}")
    return writer
