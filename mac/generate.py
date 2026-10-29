# Mac port of WanGP - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt). Added 2026-10-29.
"""Generate a video from the command line on a Mac.

    python mac/generate.py "A red fox walking through fresh snow at sunrise"
    python mac/generate.py "A cat jumps onto a sofa" --seconds 5 --preset draft --seed 42 --out ~/Movies
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if "--offline" in sys.argv:  # before any Hugging Face library loads: WanGP never downloads, a missing file is an error
    os.environ["HF_HUB_OFFLINE"] = "1"
from engine import Engine  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Text-to-video on Apple Silicon (Wan 2.1 1.3B through WanGP).")
    parser.add_argument("prompt", help="what the video shows (English works best)")
    parser.add_argument("--seconds", type=float, default=5, help="video length, up to 5 s (default 5)")
    parser.add_argument("--preset", choices=["quality", "high", "draft", "hd"], default="quality")
    parser.add_argument("--seed", type=int, default=-1, help="-1 for a random seed")
    parser.add_argument("--out", default=None, help="output folder (default: outputs/)")
    parser.add_argument("--offline", action="store_true", help="never download anything")
    args = parser.parse_args()

    out = sys.__stdout__
    engine = Engine(output_dir=Path(args.out).expanduser() if args.out else None, offline=args.offline)

    def progress(phase, step, total, elapsed):
        out.write(f"\r[{elapsed:5.0f}s] {phase} {step}/{total}   ")
        out.flush()

    path = engine.generate(args.prompt, seconds=args.seconds, preset=args.preset, seed=args.seed, on_progress=progress)
    out.write(f"\nSaved: {path}\n")


if __name__ == "__main__":
    main()
