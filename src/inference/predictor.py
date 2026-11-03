from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

from src.core.project import ProjectContext
from src.inference.types import EmotionPrediction
from src.models.registry import (
    build_model,
    get_model_spec,
)
from src.training.checkpoint import load_checkpoint


@dataclass(frozen=True)
class PredictorMetadata:
    artifact: str
    backend: str
    model_name: str
    model_family: str
    input_size: int
    num_classes: int
    device: str
    checkpoint_epoch: int = -1
    checkpoint: str | None = None
    execution_provider: str | None = None
    precision: str = "fp32"
    channels_last: bool = False
    num_threads: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class EmotionPredictor(Protocol):
    input_size: int
    class_names: tuple[str, ...]
    metadata: PredictorMetadata

    def predict_batch(
        self,
        batch: torch.Tensor,
    ) -> list[EmotionPrediction]: ...


def _validate_class_contract(
    context: ProjectContext,
) -> tuple[int, list[str]]:
    num_classes = int(
        context.config["project"]["num_classes"]
    )
    class_names = [
        str(name)
        for name in context.config["labels"]
    ]
    if len(class_names) != num_classes:
        raise RuntimeError(
            "config.yaml labels do not match project.num_classes."
        )
    return num_classes, class_names


def _validate_frozen_resolution(
    context: ProjectContext,
    input_size: int,
) -> None:
    frozen = int(
        context.config["experiments"]["p4"]["final_resolution"]
    )
    if int(input_size) != frozen:
        raise RuntimeError(
            "Inference artifact input size does not match the frozen P4 "
            f"resolution: artifact={input_size}, frozen={frozen}."
        )


def _validate_batch(batch: torch.Tensor, input_size: int) -> None:
    if batch.ndim != 4:
        raise ValueError("Expected NCHW tensor for emotion inference.")
    if int(batch.shape[1]) != 1:
        raise ValueError("Emotion model expects one grayscale channel.")
    if tuple(batch.shape[-2:]) != (input_size, input_size):
        raise ValueError(
            "Emotion batch resolution mismatch: "
            f"got {tuple(batch.shape[-2:])}, expected {(input_size, input_size)}."
        )
    if batch.dtype != torch.float32:
        raise ValueError(
            f"Emotion input must be float32 before backend casting, got {batch.dtype}."
        )


def _probability_tensor_to_predictions(
    probabilities: torch.Tensor,
    class_names: tuple[str, ...] | list[str],
) -> list[EmotionPrediction]:
    # One small device->host synchronization is intentional here because the
    # renderer/tracker consumes probabilities on the CPU immediately.
    probabilities = probabilities.detach().float().cpu()
    confidence, class_ids = probabilities.max(dim=1)
    rows = probabilities.tolist()
    confidence_values = confidence.tolist()
    class_id_values = class_ids.tolist()

    return [
        EmotionPrediction(
            class_id=int(class_id),
            class_name=str(class_names[int(class_id)]),
            confidence=float(score),
            probabilities=tuple(float(value) for value in row),
        )
        for class_id, score, row in zip(
            class_id_values,
            confidence_values,
            rows,
        )
    ]


def _probability_array_to_predictions(
    probabilities: np.ndarray,
    class_names: tuple[str, ...] | list[str],
) -> list[EmotionPrediction]:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 2:
        raise RuntimeError(
            f"Expected NxC probabilities from ONNX Runtime, got {probabilities.shape}."
        )

    class_ids = np.argmax(probabilities, axis=1)
    confidence = probabilities[
        np.arange(probabilities.shape[0]),
        class_ids,
    ]

    return [
        EmotionPrediction(
            class_id=int(class_id),
            class_name=str(class_names[int(class_id)]),
            confidence=float(score),
            probabilities=tuple(float(value) for value in row),
        )
        for class_id, score, row in zip(
            class_ids.tolist(),
            confidence.tolist(),
            probabilities.tolist(),
        )
    ]


def _numpy_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2:
        raise RuntimeError(f"Expected NxC logits, got shape {logits.shape}.")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    np.exp(shifted, out=shifted)
    denominator = np.sum(shifted, axis=1, keepdims=True)
    return shifted / np.maximum(denominator, np.finfo(np.float32).tiny)


class TorchEmotionPredictor:
    """Optimized PyTorch backend for CPU and CUDA inference.

    CPU:
      * inference_mode
      * optional thread control
      * channels-last model/input path for oneDNN-friendly convolutions

    CUDA:
      * inference_mode
      * fixed-shape cuDNN autotuning
      * channels-last
      * FP16/BF16 autocast when requested
      * non-blocking H2D transfer from pinned host memory
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        class_names: list[str],
        input_size: int,
        device: torch.device,
        metadata: PredictorMetadata,
        amp_mode: str = "auto",
        channels_last: bool = True,
        pin_memory: bool = True,
        num_threads: int | None = None,
        cudnn_benchmark: bool = True,
        allow_tf32: bool = True,
    ) -> None:
        self.class_names = tuple(class_names)
        self.input_size = int(input_size)
        self.device = device
        self.channels_last = bool(channels_last)
        self.pin_memory = bool(pin_memory and device.type == "cuda")
        self.num_threads = int(num_threads) if num_threads is not None else None
        self._amp_dtype = self._resolve_amp_dtype(device, amp_mode)

        if self.num_threads is not None:
            if self.num_threads <= 0:
                raise ValueError("num_threads must be positive.")
            if device.type == "cpu":
                torch.set_num_threads(self.num_threads)

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
            # Fixed 160x160 inference benefits from autotuning. TF32 is a
            # fallback speed path when AMP is disabled or an op remains FP32.
            if allow_tf32:
                try:
                    torch.set_float32_matmul_precision("high")
                except (AttributeError, RuntimeError):
                    pass
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except AttributeError:
                    pass

        self.model = model.to(self.device)
        if self.channels_last:
            self.model = self.model.to(
                memory_format=torch.channels_last
            )
        self.model.eval()

        precision = (
            "fp16"
            if self._amp_dtype == torch.float16
            else "bf16"
            if self._amp_dtype == torch.bfloat16
            else "fp32"
        )
        self.metadata = PredictorMetadata(
            **{
                **metadata.to_dict(),
                "precision": precision,
                "channels_last": self.channels_last,
                "num_threads": self.num_threads,
            }
        )

    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        value = value.lower().strip()
        if value == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        if value not in {"cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if value == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(value)

    @staticmethod
    def _resolve_amp_dtype(
        device: torch.device,
        mode: str,
    ) -> torch.dtype | None:
        mode = str(mode).lower().strip()
        if mode not in {"auto", "off", "fp32", "fp16", "bf16"}:
            raise ValueError(
                "amp_mode must be one of: auto, off, fp32, fp16, bf16"
            )

        if mode in {"off", "fp32"}:
            return None

        if mode == "auto":
            return torch.float16 if device.type == "cuda" else None

        if mode == "fp16":
            if device.type != "cuda":
                raise ValueError("FP16 autocast is only enabled for CUDA here.")
            return torch.float16

        if device.type == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported"):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("CUDA BF16 was requested but is unsupported.")
        return torch.bfloat16

    @classmethod
    def from_checkpoint(
        cls,
        *,
        context: ProjectContext,
        checkpoint_path: Path,
        device: str = "auto",
        amp_mode: str = "auto",
        channels_last: bool = True,
        pin_memory: bool = True,
        num_threads: int | None = None,
        cudnn_benchmark: bool = True,
        allow_tf32: bool = True,
    ) -> "TorchEmotionPredictor":
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = context.root / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Classifier checkpoint not found:\n{checkpoint_path}"
            )

        run_config_path = checkpoint_path.parent / "run_config.json"
        if not run_config_path.is_file():
            raise FileNotFoundError(
                "Inference requires run_config.json beside the project "
                f"checkpoint so model/input contracts are not guessed:\n{run_config_path}"
            )

        run_config = json.loads(
            run_config_path.read_text(encoding="utf-8")
        )
        model_name = str(run_config["model"])
        input_size = int(run_config["input_size"])
        num_classes, class_names = _validate_class_contract(context)
        _validate_frozen_resolution(context, input_size)

        model = build_model(
            model_name,
            num_classes=num_classes,
            pretrained=False,
        )
        checkpoint = load_checkpoint(checkpoint_path, model)
        resolved_device = cls._resolve_device(device)
        spec = get_model_spec(model_name)

        metadata = PredictorMetadata(
            artifact=str(checkpoint_path),
            backend="pytorch",
            checkpoint=str(checkpoint_path),
            model_name=model_name,
            model_family=spec.family,
            input_size=input_size,
            num_classes=num_classes,
            device=str(resolved_device),
            checkpoint_epoch=int(
                checkpoint.get("epoch", run_config.get("checkpoint_epoch", -1))
            ),
            execution_provider=(
                "CUDA/cuDNN"
                if resolved_device.type == "cuda"
                else "CPU/oneDNN"
            ),
        )
        del checkpoint

        return cls(
            model=model,
            class_names=class_names,
            input_size=input_size,
            device=resolved_device,
            metadata=metadata,
            amp_mode=amp_mode,
            channels_last=channels_last,
            pin_memory=pin_memory,
            num_threads=num_threads,
            cudnn_benchmark=cudnn_benchmark,
            allow_tf32=allow_tf32,
        )

    def _move_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda" and self.pin_memory and not batch.is_pinned():
            batch = batch.pin_memory()

        memory_format = (
            torch.channels_last
            if self.channels_last
            else torch.contiguous_format
        )
        return batch.to(
            self.device,
            dtype=torch.float32,
            non_blocking=self.device.type == "cuda" and batch.is_pinned(),
            memory_format=memory_format,
        )

    @torch.inference_mode()
    def predict_batch(
        self,
        batch: torch.Tensor,
    ) -> list[EmotionPrediction]:
        _validate_batch(batch, self.input_size)
        if int(batch.shape[0]) == 0:
            return []

        batch = self._move_batch(batch)
        autocast_context = (
            torch.autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
            )
            if self._amp_dtype is not None
            else nullcontext()
        )
        with autocast_context:
            logits = self.model(batch)

        probabilities = torch.softmax(logits.float(), dim=1)
        return _probability_tensor_to_predictions(
            probabilities,
            self.class_names,
        )


class ONNXEmotionPredictor:
    """ONNX Runtime backend with CPU and optional CUDA execution providers."""

    def __init__(
        self,
        *,
        session,
        class_names: list[str],
        input_size: int,
        metadata: PredictorMetadata,
    ) -> None:
        self.session = session
        self.class_names = tuple(class_names)
        self.input_size = int(input_size)
        self.metadata = metadata
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

    @staticmethod
    def _metadata_path(model_path: Path) -> Path:
        candidates = [
            model_path.with_suffix(".metadata.json"),
            model_path.parent / "metadata.json",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "ONNX metadata not found. Expected one of:\n"
            + "\n".join(str(path) for path in candidates)
        )

    @staticmethod
    def available_providers() -> tuple[str, ...]:
        try:
            import onnxruntime as ort
        except ImportError:
            return ()
        return tuple(str(value) for value in ort.get_available_providers())

    @classmethod
    def from_model(
        cls,
        *,
        context: ProjectContext,
        model_path: Path,
        num_threads: int | None = None,
        device: str = "cpu",
    ) -> "ONNXEmotionPredictor":
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "onnxruntime is required for the ONNX backend."
            ) from error

        model_path = Path(model_path).expanduser()
        if not model_path.is_absolute():
            model_path = context.root / model_path
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        metadata_path = cls._metadata_path(model_path)
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        num_classes, config_names = _validate_class_contract(context)

        model_name = str(payload["model_name"])
        input_size = int(payload["input_size"])
        artifact_num_classes = int(payload.get("num_classes", num_classes))
        class_names = [str(value) for value in payload.get("labels", config_names)]

        if artifact_num_classes != num_classes or class_names != config_names:
            raise RuntimeError(
                "ONNX metadata class contract disagrees with config.yaml."
            )
        _validate_frozen_resolution(context, input_size)
        spec = get_model_spec(model_name)

        requested_device = str(device).lower().strip()
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

        available = set(ort.get_available_providers())
        cuda_available = "CUDAExecutionProvider" in available
        if requested_device == "cuda" and not cuda_available:
            raise RuntimeError(
                "ONNX CUDA was requested, but CUDAExecutionProvider is unavailable. "
                "Install a compatible onnxruntime-gpu build or use PyTorch CUDA."
            )
        use_cuda = (
            requested_device == "cuda"
            or (requested_device == "auto" and cuda_available)
        )

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if not use_cuda and num_threads is not None:
            if num_threads <= 0:
                raise ValueError("num_threads must be positive.")
            options.intra_op_num_threads = int(num_threads)
            options.inter_op_num_threads = 1

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers,
        )
        active_provider = str(session.get_providers()[0])
        resolved_device = (
            "cuda" if active_provider == "CUDAExecutionProvider" else "cpu"
        )

        metadata = PredictorMetadata(
            artifact=str(model_path),
            backend="onnxruntime",
            checkpoint=payload.get("source_checkpoint"),
            model_name=model_name,
            model_family=spec.family,
            input_size=input_size,
            num_classes=num_classes,
            device=resolved_device,
            checkpoint_epoch=int(payload.get("checkpoint_epoch", -1)),
            execution_provider=active_provider,
            precision="fp32",
            channels_last=False,
            num_threads=(int(num_threads) if not use_cuda and num_threads else None),
        )

        return cls(
            session=session,
            class_names=class_names,
            input_size=input_size,
            metadata=metadata,
        )

    def predict_batch(
        self,
        batch: torch.Tensor,
    ) -> list[EmotionPrediction]:
        _validate_batch(batch, self.input_size)
        if int(batch.shape[0]) == 0:
            return []

        # CPU preprocessing creates a contiguous float32 tensor. .numpy() is
        # zero-copy in the normal CPU path; ORT handles the H2D transfer if a
        # CUDA execution provider was explicitly selected.
        batch_numpy = (
            batch.detach()
            .to("cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
        )
        logits = self.session.run(
            [self._output_name],
            {self._input_name: batch_numpy},
        )[0]
        probabilities = _numpy_softmax(logits)
        return _probability_array_to_predictions(
            probabilities,
            self.class_names,
        )


def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def onnx_cuda_available() -> bool:
    return "CUDAExecutionProvider" in ONNXEmotionPredictor.available_providers()


def _resolved_onnx_path(
    context: ProjectContext,
    onnx_model_path: Path,
) -> Path:
    resolved = Path(onnx_model_path).expanduser()
    if not resolved.is_absolute():
        resolved = context.root / resolved
    return resolved.resolve()


def choose_auto_backend(
    *,
    context: ProjectContext,
    onnx_model_path: Path,
    device: str,
) -> tuple[str, str]:
    """Resolve the fast default without accidentally putting GPU hosts on CPU.

    * CUDA host/request: prefer PyTorch CUDA (FP16 + channels-last fast path).
      ONNX CUDA remains available explicitly for benchmarking/deployment choice.
    * CPU host/request: prefer ONNX Runtime CPU when the artifact exists.
    """

    resolved_device = TorchEmotionPredictor._resolve_device(device)
    onnx_path = _resolved_onnx_path(context, onnx_model_path)
    onnx_exists = onnx_path.is_file() and onnxruntime_available()

    if resolved_device.type == "cuda":
        return "pytorch", "cuda"

    if onnx_exists:
        return "onnxruntime", "cpu"
    return "pytorch", "cpu"


def build_emotion_predictor(
    *,
    context: ProjectContext,
    backend: str,
    checkpoint_path: Path,
    onnx_model_path: Path,
    device: str = "auto",
    onnx_num_threads: int | None = None,
    torch_num_threads: int | None = None,
    torch_amp: str = "auto",
    channels_last: bool = True,
    pin_memory: bool = True,
    cudnn_benchmark: bool = True,
    allow_tf32: bool = True,
) -> EmotionPredictor:
    backend = backend.lower().strip()
    if backend not in {"auto", "pytorch", "onnxruntime"}:
        raise ValueError("backend must be auto, pytorch, or onnxruntime")

    resolved_device = device
    if backend == "auto":
        backend, resolved_device = choose_auto_backend(
            context=context,
            onnx_model_path=onnx_model_path,
            device=device,
        )

    if backend == "onnxruntime":
        return ONNXEmotionPredictor.from_model(
            context=context,
            model_path=onnx_model_path,
            num_threads=onnx_num_threads,
            device=resolved_device,
        )

    return TorchEmotionPredictor.from_checkpoint(
        context=context,
        checkpoint_path=checkpoint_path,
        device=resolved_device,
        amp_mode=torch_amp,
        channels_last=channels_last,
        pin_memory=pin_memory,
        num_threads=torch_num_threads,
        cudnn_benchmark=cudnn_benchmark,
        allow_tf32=allow_tf32,
    )
