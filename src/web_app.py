from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlparse

import cv2

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.core.project import ProjectContext
from src.inference.detector import YuNetFaceDetector
from src.inference.pipeline import FrameEmotionProcessor
from src.inference.predictor import build_emotion_predictor
from src.inference.renderer import render_frame_results


HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>FER Camera Monitor</title>
  <style>
    :root{color-scheme:dark;--bg:#07111f;--panel:#101d2d;--line:#253850;--accent:#45d6ad;--muted:#91a4ba}
    *{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 20% 0,#12304a 0,transparent 38%),var(--bg);font:16px system-ui;color:#eef6ff}
    main{width:min(1780px,calc(100vw - 48px));margin:24px auto}.top{display:flex;align-items:end;justify-content:space-between;gap:24px;margin-bottom:20px}
    h1{margin:0;font-size:clamp(32px,3.2vw,58px);letter-spacing:-1.5px}.eyebrow{color:var(--accent);font-weight:700;text-transform:uppercase;letter-spacing:2px;font-size:13px}
    .badge{padding:12px 18px;border:1px solid var(--line);border-radius:999px;background:#0d1928;color:var(--muted);font-size:15px}
    .grid{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:22px}.card{background:rgba(16,29,45,.92);border:1px solid var(--line);border-radius:18px;box-shadow:0 18px 55px #0007;overflow:hidden}
    .video{height:clamp(560px,calc(100vh - 190px),920px);background:#030912;display:grid;place-items:center}.video img{width:100%;height:100%;object-fit:contain}.side{padding:24px;display:grid;gap:16px;align-content:start}
    .metric{padding:22px;border:1px solid var(--line);border-radius:14px;background:#0a1624}.metric span{display:block;color:var(--muted);font-size:14px;margin-bottom:8px}.metric strong{font-size:32px}
    .source{font:14px ui-monospace;line-height:1.5;overflow-wrap:anywhere;color:#bfd0e3}.ok{color:var(--accent)}.bad{color:#ff7c82}
    footer{margin-top:16px;color:var(--muted);font-size:14px}@media(max-width:900px){main{width:min(94vw,760px)}.grid{grid-template-columns:1fr}.video{height:auto;min-height:54vw;aspect-ratio:16/9}.side{grid-template-columns:repeat(2,minmax(0,1fr))}.top{align-items:start;flex-direction:column}}@media(max-width:560px){main{width:94vw;margin:16px auto}.side{grid-template-columns:1fr}.video{min-height:58vw}.metric{padding:16px}}
  </style>
</head>
<body><main>
  <div class="top"><div><div class="eyebrow">Realtime emotion recognition</div><h1>FER Camera Monitor</h1></div><div id="state" class="badge">Đang kết nối…</div></div>
  <div class="grid"><section class="card video"><img src="/video_feed" alt="Luồng camera DroidCam"></section>
  <aside class="card side"><div class="metric"><span>FPS xử lý</span><strong id="fps">0.0</strong></div><div class="metric"><span>Khuôn mặt</span><strong id="faces">0</strong></div><div class="metric"><span>Model</span><strong id="model" style="font-size:17px">—</strong></div><div class="metric"><span>Nguồn camera</span><div id="source" class="source">—</div></div></aside></div>
  <footer>Bounding box, nhãn và confidence được vẽ trực tiếp lên từng khung hình.</footer>
</main><script>
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'}),s=await r.json();fps.textContent=s.fps.toFixed(1);faces.textContent=s.faces;model.textContent=s.model;source.textContent=s.source;state.textContent=s.connected?'Camera đang hoạt động':(s.error||'Đang kết nối…');state.className='badge '+(s.connected?'ok':'bad')}catch(e){state.textContent='Mất kết nối web';state.className='badge bad'}}
setInterval(refresh,700);refresh();
</script></body></html>"""


class CameraRuntime:
    def __init__(self, source: str, model_alias: str, backend: str = "auto") -> None:
        self.source, self.model_alias = source, model_alias
        self.capture_source = int(source) if source.isdecimal() else source
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.connected = False
        self.error = "Chưa kết nối camera"
        self.fps = 0.0
        self.faces = 0
        self.frames = 0
        self.stop_event = threading.Event()
        context = ProjectContext.load()
        family = "m1" if model_alias == "cnn" else "m2"
        checkpoint = context.root / "weights/models" / family / (
            "Notebook_CNN_best_model.pth" if family == "m1" else "EfficientNetB0_best_model.pth"
        )
        onnx_model = context.root / "weights/models" / family / (
            "Notebook_CNN.onnx" if family == "m1" else "EfficientNetB0.onnx"
        )
        predictor = build_emotion_predictor(
            context=context, backend=backend, checkpoint_path=checkpoint,
            onnx_model_path=onnx_model, device="cpu", onnx_num_threads=4,
            torch_num_threads=4,
        )
        detector = YuNetFaceDetector(
            context.root / "weights/detector/face_detection_yunet_2023mar.onnx",
            score_threshold=0.8, nms_threshold=0.3, top_k=50, max_input_side=640,
        )
        self.model_name = predictor.metadata.model_name
        self.processor = FrameEmotionProcessor(detector=detector, predictor=predictor)

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True, name="camera-worker").start()

    def _loop(self) -> None:
        capture = None
        previous = time.perf_counter()
        while not self.stop_event.is_set():
            if capture is None or not capture.isOpened():
                capture = cv2.VideoCapture(self.capture_source)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not capture.isOpened():
                    self.connected, self.error = False, (
                        "Không kết nối được DroidCam. Hãy mở app trên điện thoại "
                        "và kiểm tra IP/cổng."
                    )
                    capture.release(); time.sleep(1.0); continue
            ok, frame = capture.read()
            if not ok or frame is None:
                self.connected, self.error = False, "Mất luồng camera, đang thử lại"
                capture.release(); capture = None; time.sleep(0.5); continue
            try:
                output = self.processor.process(frame, frame_index=self.frames)
                now = time.perf_counter(); loop_fps = 1.0 / max(now - previous, 1e-6); previous = now
                self.fps = loop_fps if self.fps == 0 else 0.85 * self.fps + 0.15 * loop_fps
                rendered = render_frame_results(
                    frame, list(output.results), confidence_threshold=0.5,
                    processing_fps=output.processing_fps, stream_fps=self.fps,
                    frame_index=self.frames,
                )
                encoded, buffer = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not encoded:
                    continue
                with self.condition:
                    self.jpeg = buffer.tobytes(); self.faces = output.num_faces
                    self.frames += 1; self.connected = True; self.error = ""
                    self.condition.notify_all()
            except Exception as exc:
                self.connected, self.error = False, f"Lỗi xử lý: {exc}"
        if capture is not None:
            capture.release()

    def status(self) -> dict:
        return {"connected": self.connected, "error": self.error, "fps": self.fps,
                "faces": self.faces, "frames": self.frames, "model": self.model_name,
                "source": self.source}


class Handler(BaseHTTPRequestHandler):
    runtime: CameraRuntime

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", HTML.encode())
        elif path == "/api/status":
            self._send(200, "application/json", json.dumps(self.runtime.status()).encode())
        elif path == "/video_feed":
            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with self.runtime.condition:
                        self.runtime.condition.wait(timeout=2)
                        frame = self.runtime.jpeg
                    if frame is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self._send(404, "text/plain", b"Not found")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status); self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Local FER camera web dashboard")
    parser.add_argument(
        "--source",
        default="http://192.168.1.2:4747/video",
    )
    parser.add_argument("--model", choices=["cnn", "transfer"], default="transfer")
    parser.add_argument(
        "--backend",
        choices=["auto", "onnxruntime", "pytorch"],
        default="auto",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    runtime = CameraRuntime(args.source, args.model, args.backend)
    Handler.runtime = runtime
    runtime.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Web UI: http://{args.host}:{args.port}")
    print(f"Camera: {args.source}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop_event.set(); server.server_close()


if __name__ == "__main__":
    main()
