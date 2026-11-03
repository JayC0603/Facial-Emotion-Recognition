"""Compatibility entry point for the M1 and M2 model definitions."""

from src.models.notebook_models import NotebookEfficientNetB0, NotebookEmotionCNN
from src.models.registry import ModelSpec, build_model, count_parameters, get_model_spec

__all__ = [
    "NotebookEmotionCNN",
    "NotebookEfficientNetB0",
    "ModelSpec",
    "build_model",
    "count_parameters",
    "get_model_spec",
]

