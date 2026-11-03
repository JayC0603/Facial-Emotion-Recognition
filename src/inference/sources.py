from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, Thread
from time import monotonic, sleep
from typing import Callable, Iterator
from urllib.parse import urlparse

import cv2
import numpy as np


IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}

STREAM_SCHEMES = {
    "http",
    "https",
    "rtsp",
    "rtmp",
}


def is_stream_source(value: str) -> bool:
    parsed = urlparse(str(value))
    return parsed.scheme.lower() in STREAM_SCHEMES


def infer_source_kind(value: str | Path) -> str:
    if is_stream_source(str(value)):
        return "stream"
    return infer_local_source_kind(Path(value))


def infer_local_source_kind(path: Path) -> str:
    """Infer whether a local file should use image or video inference."""

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in IMAGE_SUFFIXES:
        return "image"

    if suffix in VIDEO_SUFFIXES:
        return "video"

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return "image"

    capture = cv2.VideoCapture(str(path))
    try:
        if capture.isOpened():
            return "video"
    finally:
        capture.release()

    raise RuntimeError(
        "Unable to identify source as an image or video file: "
        f"{path}"
    )


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int | None


class LocalVideoReader:
    """Sequential local-video reader for deterministic offline inference."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"OpenCV failed to open video: {self.path}")

        width = int(round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0.0:
            fps = 30.0

        raw_frame_count = int(round(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_count = raw_frame_count if raw_frame_count > 0 else None

        self.metadata = VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )

    def __enter__(self) -> "LocalVideoReader":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        frame_index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                break
            yield frame_index, frame
            frame_index += 1


@dataclass(frozen=True)
class StreamStatistics:
    reconnect_count: int
    frames_captured: int
    last_frame_age_seconds: float | None


class NetworkStreamReader:
    """Threaded latest-frame reader with reconnect for IP/RTSP streams.

    The capture worker continuously replaces a single latest-frame slot. A
    slow classifier therefore drops stale frames instead of accumulating
    latency. Read failures trigger bounded exponential reconnect backoff.
    """

    def __init__(
        self,
        source: str,
        *,
        reconnect_initial_seconds: float = 0.5,
        reconnect_max_seconds: float = 5.0,
        buffer_size: int = 1,
        open_timeout_seconds: float = 5.0,
        target_width: int | None = None,
        target_height: int | None = None,
        capture_factory: Callable[[str], object] | None = None,
    ) -> None:
        if not is_stream_source(source):
            raise ValueError(f"Not a supported network stream URL: {source}")
        if reconnect_initial_seconds <= 0.0:
            raise ValueError("reconnect_initial_seconds must be > 0.")
        if reconnect_max_seconds < reconnect_initial_seconds:
            raise ValueError(
                "reconnect_max_seconds must be >= reconnect_initial_seconds."
            )

        self.source = str(source)
        self.reconnect_initial_seconds = float(reconnect_initial_seconds)
        self.reconnect_max_seconds = float(reconnect_max_seconds)
        self.buffer_size = int(max(1, buffer_size))
        self.open_timeout_seconds = float(max(0.1, open_timeout_seconds))
        self.target_width = self._validate_target_dimension(
            target_width,
            name="target_width",
        )
        self.target_height = self._validate_target_dimension(
            target_height,
            name="target_height",
        )
        self._capture_factory = capture_factory or cv2.VideoCapture

        self._condition = Condition()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_sequence = -1
        self._latest_timestamp: float | None = None
        self._frames_captured = 0
        self._reconnect_count = 0
        self._capture = None
        self._worker_error: BaseException | None = None

    @staticmethod
    def _validate_target_dimension(
        value: int | None,
        *,
        name: str,
    ) -> int | None:
        if value is None:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer when provided.")
        return value

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a decoded stream frame to the requested processing size.

        OpenCV capture properties are only best-effort for HTTP/RTSP streams.
        This post-decode resize therefore guarantees the dimensions seen by the
        FER pipeline. If only width or height is supplied, preserve aspect ratio.
        """

        target_width = self.target_width
        target_height = self.target_height
        if target_width is None and target_height is None:
            return frame

        source_height, source_width = frame.shape[:2]
        if source_width <= 0 or source_height <= 0:
            raise RuntimeError("Decoded stream frame has invalid dimensions.")

        if target_width is None:
            scale = target_height / source_height
            target_width = max(1, int(round(source_width * scale)))
        elif target_height is None:
            scale = target_width / source_width
            target_height = max(1, int(round(source_height * scale)))

        assert target_width is not None
        assert target_height is not None

        if source_width == target_width and source_height == target_height:
            return frame

        downscaling = target_width < source_width or target_height < source_height
        interpolation = cv2.INTER_AREA if downscaling else cv2.INTER_LINEAR
        return cv2.resize(
            frame,
            (int(target_width), int(target_height)),
            interpolation=interpolation,
        )

    def __enter__(self) -> "NetworkStreamReader":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(
            target=self._capture_loop,
            name="fer-network-capture",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._release_capture()

    def statistics(self) -> StreamStatistics:
        with self._condition:
            age = (
                monotonic() - self._latest_timestamp
                if self._latest_timestamp is not None
                else None
            )
            return StreamStatistics(
                reconnect_count=self._reconnect_count,
                frames_captured=self._frames_captured,
                last_frame_age_seconds=age,
            )

    def frames(
        self,
        *,
        timeout_seconds: float = 10.0,
    ) -> Iterator[tuple[int, np.ndarray]]:
        self.start()
        last_sequence = -1

        while not self._stop_event.is_set():
            with self._condition:
                ready = self._condition.wait_for(
                    lambda: (
                        self._latest_sequence > last_sequence
                        or self._worker_error is not None
                        or self._stop_event.is_set()
                    ),
                    timeout=timeout_seconds,
                )

                if self._worker_error is not None:
                    error = self._worker_error
                    self._worker_error = None
                    raise RuntimeError("Network capture worker failed.") from error

                if self._stop_event.is_set():
                    break

                if not ready or self._latest_sequence <= last_sequence:
                    continue

                sequence = self._latest_sequence
                frame = self._latest_frame
                if frame is None:
                    continue
                # The worker replaces the latest-frame object; it never mutates
                # an already-published NumPy frame. Keep the object reference
                # instead of copying a full 720p/1080p frame on every consume.
                published_frame = frame

            last_sequence = sequence
            yield sequence, published_frame

    def _open_capture(self):
        capture = self._capture_factory(self.source)

        try:
            if hasattr(capture, "set"):
                capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                # Best effort only: many HTTP/RTSP sources ignore these
                # properties. _resize_frame() below guarantees the dimensions
                # actually consumed by the FER pipeline.
                if self.target_width is not None:
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                if self.target_height is not None:
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    capture.set(
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        int(self.open_timeout_seconds * 1000),
                    )
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    capture.set(
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        int(self.open_timeout_seconds * 1000),
                    )
        except Exception:
            pass

        if not capture.isOpened():
            capture.release()
            return None
        return capture

    def _release_capture(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

    def _capture_loop(self) -> None:
        backoff = self.reconnect_initial_seconds

        try:
            while not self._stop_event.is_set():
                if self._capture is None:
                    capture = self._open_capture()
                    if capture is None:
                        self._reconnect_count += 1
                        print(
                            "Khong the ket noi camera; "
                            f"dang thu lai sau {backoff:.1f} giay...",
                            flush=True,
                        )
                        if self._stop_event.wait(backoff):
                            break
                        backoff = min(
                            backoff * 2.0,
                            self.reconnect_max_seconds,
                        )
                        continue
                    self._capture = capture
                    if self._reconnect_count:
                        print("Da ket noi lai camera.", flush=True)
                    backoff = self.reconnect_initial_seconds

                ok, frame = self._capture.read()
                if not ok or frame is None:
                    self._release_capture()
                    self._reconnect_count += 1
                    print(
                        "Mat ket noi camera; "
                        f"dang thu lai sau {backoff:.1f} giay...",
                        flush=True,
                    )
                    if self._stop_event.wait(backoff):
                        break
                    backoff = min(
                        backoff * 2.0,
                        self.reconnect_max_seconds,
                    )
                    continue

                frame = self._resize_frame(frame)

                timestamp = monotonic()
                with self._condition:
                    self._latest_frame = frame
                    self._latest_sequence += 1
                    self._latest_timestamp = timestamp
                    self._frames_captured += 1
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._worker_error = error
                self._condition.notify_all()
        finally:
            self._release_capture()
