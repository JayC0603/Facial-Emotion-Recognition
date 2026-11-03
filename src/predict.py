"""Compatibility entry point for FER preprocessing and prediction."""

from src.inference.pipeline import FrameEmotionProcessor
from src.inference.predictor import (
    EmotionPredictor,
    ONNXEmotionPredictor,
    TorchEmotionPredictor,
    build_emotion_predictor,
)
from src.inference.preprocess import InferencePreprocessor, square_face_crop
from src.inference.types import EmotionPrediction, FaceEmotionResult

__all__ = [
    "EmotionPrediction",
    "EmotionPredictor",
    "FaceEmotionResult",
    "FrameEmotionProcessor",
    "InferencePreprocessor",
    "ONNXEmotionPredictor",
    "TorchEmotionPredictor",
    "build_emotion_predictor",
    "square_face_crop",
]

