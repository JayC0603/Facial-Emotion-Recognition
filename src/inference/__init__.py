"""Reusable inference components for image, video and realtime FER deployment."""

from src.inference.detector import YuNetFaceDetector
from src.inference.pipeline import (
    FrameEmotionProcessor,
    FrameInferenceOutput,
    StaticImageEmotionPipeline,
    StaticImageInferenceOutput,
    StreamEmotionPipeline,
    StreamInferenceOutput,
    VideoEmotionPipeline,
    VideoInferenceOutput,
)
from src.inference.predictor import (
    EmotionPredictor,
    ONNXEmotionPredictor,
    PredictorMetadata,
    TorchEmotionPredictor,
    build_emotion_predictor,
)
from src.inference.preprocess import (
    InferencePreprocessor,
    square_face_crop,
)
from src.inference.sources import (
    LocalVideoReader,
    NetworkStreamReader,
    StreamStatistics,
    VideoMetadata,
    infer_local_source_kind,
    infer_source_kind,
    is_stream_source,
)
from src.inference.tracking import (
    TemporalEmotionStabilizer,
    intersection_over_union,
)
from src.inference.types import (
    EmotionPrediction,
    FaceDetection,
    FaceEmotionResult,
)

__all__ = [
    "EmotionPrediction",
    "EmotionPredictor",
    "FaceDetection",
    "FaceEmotionResult",
    "FrameEmotionProcessor",
    "FrameInferenceOutput",
    "InferencePreprocessor",
    "LocalVideoReader",
    "NetworkStreamReader",
    "ONNXEmotionPredictor",
    "PredictorMetadata",
    "StaticImageEmotionPipeline",
    "StaticImageInferenceOutput",
    "StreamEmotionPipeline",
    "StreamInferenceOutput",
    "StreamStatistics",
    "TemporalEmotionStabilizer",
    "TorchEmotionPredictor",
    "VideoEmotionPipeline",
    "VideoInferenceOutput",
    "VideoMetadata",
    "YuNetFaceDetector",
    "build_emotion_predictor",
    "infer_local_source_kind",
    "infer_source_kind",
    "intersection_over_union",
    "is_stream_source",
    "square_face_crop",
]
