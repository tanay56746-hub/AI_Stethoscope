#!/usr/bin/env python3
# ==============================================================================
# [LAYER: PRETRAINED WEIGHTS LOADER] | File: backend/download_weights.py
# Purpose: Downloads official PANNs CNN14 weights from Zenodo during Render build
# ==============================================================================
"""
AI Stethoscope - Official PANNs CNN14 Weight Downloader
=======================================================
Downloads official AudioSet pretrained weights (327.4 MB) from Zenodo into
`pretrained_weights/Cnn14_mAP=0.431.pth` if not already present.
"""

import sys
import os
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PROJECT_ROOT / "pretrained_weights"
WEIGHTS_FILE = WEIGHTS_DIR / "Cnn14_mAP=0.431.pth"
EXPECTED_SIZE = 327428481  # 312.26 MB

# URLs to attempt downloading from, in order
MIRRORS = [
    "https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth",
    "https://huggingface.co/thelou1s/panns-inference/resolve/main/Cnn14_mAP=0.431.pth"
]

def ensure_weights(destination: Path = WEIGHTS_FILE) -> Path:
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size >= EXPECTED_SIZE:
        print(f"[Weights] Official PANNs CNN14 checkpoint already verified at:")
        print(f"          {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
        return destination

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "*/*",
    }
    
    last_exc = None
    temp_dest = destination.with_suffix(".tmp")
    
    for url in MIRRORS:
        print("=" * 65)
        print("[Weights] Downloading official PANNs CNN14 checkpoint...")
        print(f"          Source: {url}")
        print(f"          Destination: {destination}")
        print("=" * 65)

        req = urllib.request.Request(url, headers=headers)
        
        try:
            start_time = time.time()
            with urllib.request.urlopen(req, timeout=60) as response, open(temp_dest, "wb") as f_out:
                total_size = int(response.headers.get("Content-Length", EXPECTED_SIZE))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1 MB chunks
                last_pct = 0

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    downloaded += len(chunk)
                    pct = int(downloaded / total_size * 100)
                    if pct >= last_pct + 10 or downloaded == total_size:
                        elapsed = max(0.1, time.time() - start_time)
                        speed_mb = (downloaded / 1e6) / elapsed
                        print(f"          Downloaded {downloaded / 1e6:.1f} MB / {total_size / 1e6:.1f} MB ({pct}%) @ {speed_mb:.1f} MB/s", flush=True)
                        last_pct = pct

            if temp_dest.stat().st_size < 1000000:
                raise ValueError(f"Downloaded file is too small ({temp_dest.stat().st_size} bytes).")

            if destination.exists():
                destination.unlink()
            temp_dest.rename(destination)
            print(f"[Weights] Download complete and verified successfully ({destination.stat().st_size / 1e6:.1f} MB).", flush=True)
            return destination

        except Exception as exc:
            print(f"[Weights] ERROR with mirror {url}: {exc}", file=sys.stderr, flush=True)
            last_exc = exc
            if temp_dest.exists():
                temp_dest.unlink()
            print("[Weights] Retrying with next mirror if available...", flush=True)
            continue
            
    print("[Weights] urllib downloads failed. Falling back to curl...", flush=True)
    import subprocess
    for url in MIRRORS:
        try:
            print(f"          curl downloading from: {url}")
            subprocess.run([
                "curl", "-L", 
                "-A", headers["User-Agent"],
                "-o", str(temp_dest), 
                url
            ], check=True, stdout=sys.stdout, stderr=sys.stderr)
            
            if temp_dest.exists() and temp_dest.stat().st_size >= EXPECTED_SIZE:
                if destination.exists():
                    destination.unlink()
                temp_dest.rename(destination)
                print(f"[Weights] curl download complete ({destination.stat().st_size / 1e6:.1f} MB).", flush=True)
                return destination
        except Exception as e:
            print(f"[Weights] curl failed for {url}: {e}", file=sys.stderr)
            if temp_dest.exists():
                temp_dest.unlink()
            
    raise RuntimeError(f"All mirrors and fallback methods failed. Last error: {last_exc}")

if __name__ == "__main__":
    ensure_weights()
