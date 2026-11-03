"""Compatibility entry point for face detection."""

from src.inference.detector import YuNetFaceDetector
from src.inference.types import FaceDetection

__all__ = ["FaceDetection", "YuNetFaceDetector"]

