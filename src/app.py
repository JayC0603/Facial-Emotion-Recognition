from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2


if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[1]
    repository_root_text = str(repository_root)
    if repository_root_text not in sys.path:
        sys.path.insert(0, repository_root_text)


from src.core.project import ProjectContext
from src.inference import (
    StaticImageEmotionPipeline,
    StreamEmotionPipeline,
    TemporalEmotionStabilizer,
    VideoEmotionPipeline,
    YuNetFaceDetector,
    build_emotion_predictor,
    infer_source_kind,
)
from src.models.registry import get_model_spec


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FER P8.6 inference: image/video/IP-stream YuNet detection, "
            "batched FER, optional temporal EMA, ONNX Runtime or PyTorch."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Image path, video path, or HTTP/HTTPS/RTSP/RTMP camera URL.",
    )
    parser.add_argument(
        "--source-kind",
        choices=["auto", "image", "video", "stream"],
        default="auto",
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "transfer"],
        default="transfer",
        help="Rubric-facing M1/M2 model family.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "onnxruntime", "pytorch"],
        default="auto",
        help=(
            "auto chooses an accelerated GPU backend on CUDA hosts and prefers "
            "ONNX Runtime on CPU when an ONNX artifact exists."
        ),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--onnx-model", default=None)
    parser.add_argument("--onnx-threads", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument(
        "--torch-amp",
        choices=["auto", "off", "fp32", "fp16", "bf16"],
        default=None,
        help="PyTorch precision policy; auto uses FP16 on CUDA and FP32 on CPU.",
    )
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use channels-last memory format for PyTorch convolution inference.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pin CPU face batches before CUDA transfer.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable cuDNN fixed-shape autotuning on CUDA.",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow TF32 fallback kernels on supported NVIDIA GPUs.",
    )
    parser.add_argument("--opencv-threads", type=int, default=None)
    parser.add_argument(
        "--detector-model",
        default="weights/detector/face_detection_yunet_2023mar.onnx",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Rendered image/video path. Stream output is optional; when set, "
            "the stream is recorded."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Inference device. PyTorch supports CPU/CUDA; ONNX Runtime also uses "
            "CUDAExecutionProvider when a compatible onnxruntime-gpu build exists."
        ),
    )
    parser.add_argument("--face-threshold", type=float, default=None)
    parser.add_argument("--face-nms-threshold", type=float, default=None)
    parser.add_argument(
        "--detector-max-side",
        type=int,
        default=None,
        help=(
            "Downscale only the YuNet detector input so its longest side is at most "
            "this many pixels. Bounding boxes are mapped back to the original frame. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument("--crop-padding", type=float, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--video-codec", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display local video. Streams display by default unless --no-display.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable preview window (useful for headless stream recording).",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Disable IoU tracking and probability EMA for video/stream.",
    )
    parser.add_argument("--ema-alpha", type=float, default=None)
    parser.add_argument("--track-iou-threshold", type=float, default=None)
    parser.add_argument("--track-max-missed", type=int, default=None)
    parser.add_argument("--reconnect-initial", type=float, default=None)
    parser.add_argument("--reconnect-max", type=float, default=None)
    parser.add_argument("--stream-timeout", type=float, default=None)
    parser.add_argument("--record-fps", type=float, default=None)
    parser.add_argument(
        "--stream-width",
        type=int,
        default=None,
        help=(
            "Target width for network-stream frames before FER processing. "
            "OpenCV also requests this width from the camera, but HTTP/RTSP "
            "servers may ignore that request; the app then resizes locally. "
            "If only width is set, aspect ratio is preserved."
        ),
    )
    parser.add_argument(
        "--stream-height",
        type=int,
        default=None,
        help=(
            "Target height for network-stream frames before FER processing. "
            "If only height is set, aspect ratio is preserved."
        ),
    )

    return parser.parse_args()


def resolve_project_path(
    context: ProjectContext,
    value: str | Path,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = context.root / path
    return path.resolve()


def default_checkpoint(
    context: ProjectContext,
    model_alias: str,
) -> Path:
    relative = (
        "weights/models/m1/Notebook_CNN_best_model.pth"
        if model_alias == "cnn"
        else "weights/models/m2/EfficientNetB0_best_model.pth"
    )
    return resolve_project_path(context, relative)


def default_onnx_model(
    context: ProjectContext,
    model_alias: str,
) -> Path:
    relative = (
        "weights/models/m1/Notebook_CNN.onnx"
        if model_alias == "cnn"
        else "weights/models/m2/EfficientNetB0.onnx"
    )
    return resolve_project_path(context, relative)


def default_output_path(
    context: ProjectContext,
    source: str,
    source_kind: str,
) -> Path | None:
    if source_kind == "stream":
        return None

    source_path = Path(source)
    suffix = ".jpg" if source_kind == "image" else ".mp4"
    return (
        context.root
        / "outputs"
        / f"{source_path.stem}_pred{suffix}"
    ).resolve()


def validate_model_alias(predictor, alias: str) -> None:
    spec = get_model_spec(predictor.metadata.model_name)
    expected_family = "custom_cnn" if alias == "cnn" else "transfer"
    if spec.family != expected_family:
        raise RuntimeError(
            "--model alias disagrees with inference artifact architecture: "
            f"alias={alias}, model={spec.name}, family={spec.family}."
        )


def build_stabilizer(
    predictor,
    runtime_cfg: dict,
    args: argparse.Namespace,
):
    if args.no_temporal:
        return None

    return TemporalEmotionStabilizer(
        class_names=list(predictor.class_names),
        iou_threshold=float(
            args.track_iou_threshold
            if args.track_iou_threshold is not None
            else runtime_cfg.get("tracking_iou_threshold", 0.30)
        ),
        max_missed_frames=int(
            args.track_max_missed
            if args.track_max_missed is not None
            else runtime_cfg.get("tracking_max_missed_frames", 10)
        ),
        ema_alpha=float(
            args.ema_alpha
            if args.ema_alpha is not None
            else runtime_cfg.get("emotion_ema_alpha", 0.55)
        ),
    )


def common_runtime_payload(
    *,
    predictor,
    detector_model_path: Path,
    face_threshold: float,
    face_nms_threshold: float,
    crop_padding: float,
    detector_max_side: int | None,
    opencv_threads: int | None,
) -> dict:
    return {
        "classifier": {
            "backend": predictor.metadata.backend,
            "artifact": predictor.metadata.artifact,
            "checkpoint": predictor.metadata.checkpoint,
            "model_name": predictor.metadata.model_name,
            "model_family": predictor.metadata.model_family,
            "checkpoint_epoch": predictor.metadata.checkpoint_epoch,
            "input_size": predictor.metadata.input_size,
            "device": predictor.metadata.device,
            "execution_provider": predictor.metadata.execution_provider,
            "precision": predictor.metadata.precision,
            "channels_last": predictor.metadata.channels_last,
            "num_threads": predictor.metadata.num_threads,
        },
        "detector": {
            "name": "yunet",
            "model": str(detector_model_path),
            "score_threshold": face_threshold,
            "nms_threshold": face_nms_threshold,
            "crop_padding_ratio": crop_padding,
            "max_input_side": detector_max_side,
        },
        "runtime_optimization": {
            "opencv_threads": opencv_threads,
        },
    }


def main() -> None:
    args = parse_arguments()
    context = ProjectContext.load()
    runtime_cfg = context.config.get("runtime", {})

    source_kind = (
        infer_source_kind(args.source)
        if args.source_kind == "auto"
        else args.source_kind
    )

    if args.stream_width is not None and args.stream_width <= 0:
        raise ValueError("--stream-width must be a positive integer.")
    if args.stream_height is not None and args.stream_height <= 0:
        raise ValueError("--stream-height must be a positive integer.")
    if source_kind != "stream" and (
        args.stream_width is not None or args.stream_height is not None
    ):
        raise ValueError(
            "--stream-width/--stream-height are only valid for network streams."
        )

    if source_kind in {"image", "video"}:
        local_source = resolve_project_path(context, args.source)
        if not local_source.is_file():
            raise FileNotFoundError(f"Local source not found: {local_source}")
        source_value = str(local_source)
    else:
        source_value = str(args.source)

    checkpoint_path = (
        resolve_project_path(context, args.checkpoint)
        if args.checkpoint is not None
        else default_checkpoint(context, args.model)
    )
    onnx_model_path = (
        resolve_project_path(context, args.onnx_model)
        if args.onnx_model is not None
        else default_onnx_model(context, args.model)
    )
    detector_model_path = resolve_project_path(context, args.detector_model)

    default_output = default_output_path(context, source_value, source_kind)
    output_path = (
        resolve_project_path(context, args.output)
        if args.output is not None
        else default_output
    )

    face_threshold = float(
        args.face_threshold
        if args.face_threshold is not None
        else runtime_cfg.get("face_detection_score_threshold", 0.8)
    )
    face_nms_threshold = float(
        args.face_nms_threshold
        if args.face_nms_threshold is not None
        else runtime_cfg.get("face_nms_threshold", 0.3)
    )
    crop_padding = float(
        args.crop_padding
        if args.crop_padding is not None
        else runtime_cfg.get("face_crop_padding_ratio", 0.10)
    )
    confidence_threshold = float(
        args.confidence_threshold
        if args.confidence_threshold is not None
        else runtime_cfg.get("confidence_threshold", 0.5)
    )
    video_codec = str(
        args.video_codec
        if args.video_codec is not None
        else runtime_cfg.get("video_codec", "mp4v")
    )
    onnx_threads = (
        args.onnx_threads
        if args.onnx_threads is not None
        else runtime_cfg.get("onnx_intra_op_threads")
    )
    torch_threads = (
        args.torch_threads
        if args.torch_threads is not None
        else runtime_cfg.get("torch_num_threads")
    )
    torch_amp = str(
        args.torch_amp
        if args.torch_amp is not None
        else runtime_cfg.get("torch_amp", "auto")
    )
    channels_last = bool(
        args.channels_last
        if args.channels_last is not None
        else runtime_cfg.get("torch_channels_last", True)
    )
    pin_memory = bool(
        args.pin_memory
        if args.pin_memory is not None
        else runtime_cfg.get("cuda_pin_memory", True)
    )
    cudnn_benchmark = bool(
        args.cudnn_benchmark
        if args.cudnn_benchmark is not None
        else runtime_cfg.get("cuda_cudnn_benchmark", True)
    )
    allow_tf32 = bool(
        args.allow_tf32
        if args.allow_tf32 is not None
        else runtime_cfg.get("cuda_allow_tf32", True)
    )
    opencv_threads = int(
        args.opencv_threads
        if args.opencv_threads is not None
        else runtime_cfg.get("opencv_num_threads", 4)
    )
    if opencv_threads <= 0:
        raise ValueError("OpenCV thread count must be positive.")
    cv2.setNumThreads(opencv_threads)

    raw_detector_max_side = (
        args.detector_max_side
        if args.detector_max_side is not None
        else runtime_cfg.get("face_detector_max_side", 640)
    )
    detector_max_side = (
        None
        if raw_detector_max_side is None or int(raw_detector_max_side) == 0
        else int(raw_detector_max_side)
    )
    if detector_max_side is not None and detector_max_side <= 0:
        raise ValueError("detector max side must be positive or 0 to disable.")

    predictor = build_emotion_predictor(
        context=context,
        backend=args.backend,
        checkpoint_path=checkpoint_path,
        onnx_model_path=onnx_model_path,
        device=args.device,
        onnx_num_threads=(
            int(onnx_threads)
            if onnx_threads is not None
            else None
        ),
        torch_num_threads=(
            int(torch_threads)
            if torch_threads is not None
            else None
        ),
        torch_amp=torch_amp,
        channels_last=channels_last,
        pin_memory=pin_memory,
        cudnn_benchmark=cudnn_benchmark,
        allow_tf32=allow_tf32,
    )
    validate_model_alias(predictor, args.model)

    detector = YuNetFaceDetector(
        detector_model_path,
        score_threshold=face_threshold,
        nms_threshold=face_nms_threshold,
        top_k=int(runtime_cfg.get("face_top_k", 50)),
        max_input_side=detector_max_side,
    )

    stabilizer = None
    if source_kind in {"video", "stream"}:
        stabilizer = build_stabilizer(predictor, runtime_cfg, args)

    if source_kind == "image":
        if output_path is None:
            raise RuntimeError("Static image inference requires an output path.")
        pipeline = StaticImageEmotionPipeline(
            detector=detector,
            predictor=predictor,
            crop_padding_ratio=crop_padding,
            confidence_threshold=confidence_threshold,
        )
        result = pipeline.run(
            source_path=Path(source_value),
            output_path=output_path,
        )
        payload = result.to_dict()
        payload["source_kind"] = "image"

    elif source_kind == "video":
        if output_path is None:
            raise RuntimeError("Local video inference requires an output path.")
        pipeline = VideoEmotionPipeline(
            detector=detector,
            predictor=predictor,
            crop_padding_ratio=crop_padding,
            confidence_threshold=confidence_threshold,
            output_codec=video_codec,
            stabilizer=stabilizer,
        )
        result = pipeline.run(
            source_path=Path(source_value),
            output_path=output_path,
            max_frames=args.max_frames,
            display=bool(args.display and not args.no_display),
        )
        payload = result.to_dict()
        payload["source_kind"] = "video"
        payload["video_codec"] = video_codec

    elif source_kind == "stream":
        display = not args.no_display
        if args.display:
            display = True
        pipeline = StreamEmotionPipeline(
            detector=detector,
            predictor=predictor,
            crop_padding_ratio=crop_padding,
            confidence_threshold=confidence_threshold,
            output_codec=video_codec,
            stabilizer=stabilizer,
            reconnect_initial_seconds=float(
                args.reconnect_initial
                if args.reconnect_initial is not None
                else runtime_cfg.get("stream_reconnect_initial_seconds", 0.5)
            ),
            reconnect_max_seconds=float(
                args.reconnect_max
                if args.reconnect_max is not None
                else runtime_cfg.get("stream_reconnect_max_seconds", 5.0)
            ),
            stream_timeout_seconds=float(
                args.stream_timeout
                if args.stream_timeout is not None
                else runtime_cfg.get("stream_frame_timeout_seconds", 10.0)
            ),
            record_fps=float(
                args.record_fps
                if args.record_fps is not None
                else runtime_cfg.get("stream_record_fps", 30.0)
            ),
            stream_width=args.stream_width,
            stream_height=args.stream_height,
        )
        result = pipeline.run(
            source=source_value,
            output_path=output_path,
            max_frames=args.max_frames,
            display=display,
        )
        payload = result.to_dict()
        payload["source_kind"] = "stream"
        payload["video_codec"] = video_codec
        payload["stream_frame_size"] = {
            "requested_width": args.stream_width,
            "requested_height": args.stream_height,
            "aspect_ratio_preserved_when_single_dimension": True,
        }

    else:
        raise RuntimeError(f"Unsupported source kind: {source_kind}")

    payload.update(
        common_runtime_payload(
            predictor=predictor,
            detector_model_path=detector_model_path,
            face_threshold=face_threshold,
            face_nms_threshold=face_nms_threshold,
            crop_padding=crop_padding,
            detector_max_side=detector_max_side,
            opencv_threads=opencv_threads,
        )
    )

    if output_path is not None:
        summary_path = output_path.with_suffix(".json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    else:
        summary_path = None

    print(json.dumps(payload, indent=2))
    if output_path is not None:
        print(f"\nRendered: {output_path}")
    if summary_path is not None:
        print(f"Summary : {summary_path}")
    if getattr(result, "frames_jsonl_path", None):
        print(f"Frames  : {result.frames_jsonl_path}")


if __name__ == "__main__":
    main()
