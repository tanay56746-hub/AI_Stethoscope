#!/usr/bin/env python3
# ==============================================================================
# [LAYER: BACKEND API SERVER] | File: backend/demo_server.py
# Purpose: HTTP REST API Server (Port 8000) connecting Frontend UI to ML Backend
# ==============================================================================
"""
AI Stethoscope - Demonstration API Server
=========================================
Lightweight, zero-dependency HTTP API server bridging the frontend demonstration
interface to the verified offline inference pipeline.

Endpoints:
  GET  /api/status      - Model readiness status and verified technical specifications
  GET  /api/samples     - List of pre-loaded sample audio recordings for quick testing
  POST /api/predict     - Upload WAV audio file or specify sample path for analysis
  GET  /                - Serves frontend build if available (frontend/dist)

Usage:
  python backend/demo_server.py [--port 8000]
"""

import sys
import os
import json
import argparse
import tempfile
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference_engine import StethoscopeInferenceEngine

# Demo samples for one-click professor testing
SAMPLE_AUDIOS = [
    {
        "id": "b0032",
        "name": "b0032.wav (Heart Sound - PhysioNet Challenge)",
        "path": "datasets/Heart/Raw/training/training-b/b0032.wav",
        "description": "5.0-second cardiac recording from training-b set",
    }
]


class StethoscopeRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler with REST API routing and CORS support."""

    def __init__(self, *args, **kwargs):
        self.frontend_dist = PROJECT_ROOT / "frontend" / "dist"
        super().__init__(*args, directory=str(self.frontend_dist if self.frontend_dist.exists() else PROJECT_ROOT), **kwargs)

    def end_headers(self):
        cors_config = os.environ.get("CORS_ALLOWED_ORIGINS", "*").strip()
        req_origin = self.headers.get("Origin", "")

        if cors_config == "*":
            allowed_origin = "*"
        elif req_origin:
            allowed_list = [o.strip() for o in cors_config.split(",") if o.strip()]
            if req_origin in allowed_list or "*" in allowed_list:
                allowed_origin = req_origin
            else:
                allowed_origin = allowed_list[0] if allowed_list else "*"
        else:
            allowed_origin = cors_config.split(",")[0].strip() if cors_config else "*"

        self.send_header("Access-Control-Allow-Origin", allowed_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept, Authorization")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def send_json(self, status_code: int, data: dict):
        response_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def send_error_json(self, status_code: int, message: str, detail: str = ""):
        payload = {
            "success": False,
            "error": message,
            "detail": detail,
        }
        self.send_json(status_code, payload)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.handle_status()
            return
        elif path == "/api/samples":
            self.handle_samples()
            return

        # Serve frontend static assets if built
        if self.frontend_dist.exists():
            file_path = self.frontend_dist / path.lstrip("/")
            if file_path.is_file():
                return super().do_GET()
            else:
                # SPA fallback to index.html
                index_path = self.frontend_dist / "index.html"
                if index_path.exists():
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    content = index_path.read_bytes()
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return

        # Default fallback
        self.send_json(404, {"error": f"Endpoint not found: {path}"})

    def handle_status(self):
        """Return model readiness status and verified technical specifications."""
        is_ready = False
        error_msg = None
        checkpoint_name = "N/A"
        panns_name = "N/A"

        try:
            engine = StethoscopeInferenceEngine.get_instance()
            panns_ok = hasattr(engine, "panns_model") and engine.panns_model is not None
            multitask_ok = hasattr(engine, "multitask_model") and engine.multitask_model is not None
            is_ready = bool(panns_ok and multitask_ok)
            if is_ready:
                checkpoint_name = engine.checkpoint_path.name
                panns_name = engine.panns_weights_path.name
        except Exception as exc:
            is_ready = False
            error_msg = str(exc)

        response = {
            "status": "ready" if is_ready else "error",
            "model_ready": is_ready,
            "error": error_msg,
            "system_info": {
                "model_name": "Shared Multitask CNN",
                "audio_format": "WAV",
                "sample_rate_hz": 16000,
                "analysis_window_sec": 5.0,
                "input_type": "WAV Audio",
            },
            "technical_specs": {
                "architecture": "Shared Multitask CNN + Physical Acoustic Features + PANNs CNN14 Embeddings",
                "log_mel_dimensions": "128 × 313",
                "acoustic_features_dim": 397,
                "panns_embedding_dim": "2048 dimensions",
                "panns_projection": "2048 → 128",
                "audio_sample_rate": "16 kHz",
                "analysis_window": "5 seconds",
                "optimizer": "AdamW",
                "checkpoint": f"experiments/multitask_ablations/cnn_physical_panns/checkpoints/{checkpoint_name}",
                "panns_weights": panns_name,
                "inference_mode": "evaluation / no gradient",
            },
        }
        self.send_json(200 if is_ready else 503, response)

    def handle_samples(self):
        """Return verified available demo sample WAV recordings."""
        available_samples = []
        for s in SAMPLE_AUDIOS:
            full_path = PROJECT_ROOT / s["path"]
            if full_path.exists():
                available_samples.append({
                    "id": s["id"],
                    "name": s["name"],
                    "path": s["path"],
                    "filename": full_path.name,
                    "size_bytes": full_path.stat().st_size,
                    "description": s["description"],
                })
        self.send_json(200, {"success": True, "samples": available_samples})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/predict":
            self.handle_predict()
            return

        self.send_json(404, {"error": f"Endpoint not found: {path}"})

    def handle_predict(self):
        """Execute inference on uploaded file or specified sample path."""
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length == 0:
            self.send_error_json(
                400,
                "Analysis could not be completed. No audio data was received in the request."
            )
            return

        temp_file_path = None
        target_audio_path = None

        try:
            if "application/json" in content_type:
                body_bytes = self.rfile.read(content_length)
                req_data = json.loads(body_bytes.decode("utf-8"))

                # Check if sample path provided
                if "sample_path" in req_data:
                    rel_path = req_data["sample_path"]
                    candidate = (PROJECT_ROOT / rel_path).resolve()
                    if not candidate.exists() or not candidate.is_file():
                        self.send_error_json(
                            404,
                            f"Sample audio file not found on server: {rel_path}"
                        )
                        return
                    target_audio_path = candidate
                elif "audio_base64" in req_data:
                    import base64
                    raw_bytes = base64.b64decode(req_data["audio_base64"])
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp.write(raw_bytes)
                    tmp.close()
                    temp_file_path = Path(tmp.name)
                    target_audio_path = temp_file_path
                else:
                    self.send_error_json(
                        400,
                        "Analysis could not be completed. Please provide a valid audio file or sample path."
                    )
                    return

            elif "multipart/form-data" in content_type:
                # Parse multipart body
                boundary = content_type.split("boundary=")[-1].encode("utf-8")
                raw_body = self.rfile.read(content_length)

                parts = raw_body.split(b"--" + boundary)
                file_bytes = None
                filename = "uploaded_audio.wav"

                for part in parts:
                    if b'filename="' in part:
                        headers_and_body = part.split(b"\r\n\r\n", 1)
                        if len(headers_and_body) == 2:
                            header_part, body_part = headers_and_body
                            # strip trailing CRLF
                            if body_part.endswith(b"\r\n"):
                                body_part = body_part[:-2]
                            file_bytes = body_part
                            # extract filename
                            for line in header_part.decode("latin1", errors="ignore").splitlines():
                                if 'filename="' in line:
                                    fn = line.split('filename="')[1].split('"')[0]
                                    if fn:
                                        filename = fn
                            break

                if not file_bytes:
                    self.send_error_json(
                        400,
                        "Analysis could not be completed. Could not parse uploaded WAV audio."
                    )
                    return

                if not filename.lower().endswith(".wav"):
                    self.send_error_json(
                        400,
                        "Invalid file format. Please upload a standard WAV (.wav) audio recording."
                    )
                    return

                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(file_bytes)
                tmp.close()
                temp_file_path = Path(tmp.name)
                target_audio_path = temp_file_path

            elif "audio/wav" in content_type or "audio/x-wav" in content_type or "application/octet-stream" in content_type:
                audio_bytes = self.rfile.read(content_length)
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(audio_bytes)
                tmp.close()
                temp_file_path = Path(tmp.name)
                target_audio_path = temp_file_path

            else:
                self.send_error_json(
                    415,
                    f"Unsupported content type '{content_type}'. Please send WAV audio."
                )
                return

            # Execute inference using cached engine
            engine = StethoscopeInferenceEngine.get_instance()
            results = engine.analyze_audio(target_audio_path)

            self.send_json(200, results)

        except Exception as exc:
            print(f"[Error] Inference error: {exc}", file=sys.stderr)
            self.send_error_json(
                500,
                "Analysis could not be completed. Please check that the audio file is a valid WAV recording.",
                detail=str(exc)
            )

        finally:
            # Clean up temporary upload file if created
            if temp_file_path and temp_file_path.exists():
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass


def run_server(port: int = 8000, host: str = "127.0.0.1"):
    server_address = (host, port)
    # Warm up inference engine so models are preloaded before first user click
    print("=" * 60, flush=True)
    print("[Render] Starting AI Stethoscope Python backend", flush=True)
    print("=" * 60, flush=True)
    print("[Render] Loading inference engine...", flush=True)
    engine = StethoscopeInferenceEngine.get_instance()
    print(f"[Render] PANNs checkpoint loaded: {engine.panns_weights_path.name}", flush=True)
    print(f"[Render] Final checkpoint loaded: {engine.checkpoint_path.name}", flush=True)
    print("[Render] Models loaded successfully.", flush=True)

    httpd = HTTPServer(server_address, StethoscopeRequestHandler)
    print(f"[Render] Server started and running on {host}:{port}", flush=True)
    print("[Render] Ready to accept incoming clinical audio requests.", flush=True)
    print("=" * 60, flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    default_port = int(os.environ.get("PORT", os.environ.get("BACKEND_PORT", "8000")))
    default_host = "0.0.0.0" if "PORT" in os.environ else os.environ.get("BACKEND_HOST", "127.0.0.1")
    parser = argparse.ArgumentParser(description="AI Stethoscope Demonstration Server")
    parser.add_argument("--port", type=int, default=default_port, help=f"Server port (default: {default_port})")
    parser.add_argument("--host", type=str, default=default_host, help=f"Host address (default: {default_host})")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)
