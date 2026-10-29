# Mac port of WanGP - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt). Added 2026-10-29.
"""A small local video service for other apps: one JSON object per line on stdin, one per line on stdout.

Requests:
    {"id": "1", "cmd": "generate", "prompt": "...", "seconds": 5, "preset": "quality", "seed": -1}
    {"id": "2", "cmd": "ping"}
Replies (same id):
    {"id": "1", "event": "progress", "phase": "inference", "step": 4, "total": 20, "elapsed": 210.5}
    {"id": "1", "event": "done", "path": "/.../video.mp4", "seconds": 702.1}
    {"id": "1", "event": "error", "message": "..."}
The model loads on the first request and stays in memory. Nothing leaves the computer (pass --offline to forbid downloads).
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The JSON replies get a private copy of stdout; everything else that writes to stdout (WanGP, C libraries) goes to stderr.
OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)


def send(message):
    OUT.write(json.dumps(message) + "\n")
    OUT.flush()


def main():
    offline = "--offline" in sys.argv
    output_dir = None
    if "--output-dir" in sys.argv:
        output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1]).expanduser()
    engine = None
    send({"event": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            send({"event": "error", "message": "invalid JSON"})
            continue
        rid = request.get("id")
        cmd = request.get("cmd")
        if cmd == "ping":
            send({"id": rid, "event": "pong"})
            continue
        if cmd != "generate":
            send({"id": rid, "event": "error", "message": f"unknown command {cmd!r}"})
            continue
        try:
            if engine is None:
                send({"id": rid, "event": "progress", "phase": "loading", "step": 0, "total": 1, "elapsed": 0})
                from engine import Engine
                engine = Engine(output_dir=output_dir, offline=offline, log=lambda text: print(text, file=sys.stderr))
            start = time.time()
            path = engine.generate(
                request.get("prompt", ""), seconds=float(request.get("seconds", 5)), preset=request.get("preset", "quality"),
                seed=int(request.get("seed", -1)),
                on_progress=lambda phase, step, total, elapsed: send({"id": rid, "event": "progress", "phase": phase, "step": step,
                                                                      "total": total, "elapsed": round(elapsed, 1)}))
            send({"id": rid, "event": "done", "path": str(path), "seconds": round(time.time() - start, 1)})
        except Exception as error:  # report and keep serving
            send({"id": rid, "event": "error", "message": str(error)})


if __name__ == "__main__":
    main()
