# Mac port of WanGP - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt). Added 2026-10-29.
"""Downloads only the files that text-to-video with Wan 2.1 1.3B needs (about 10 GB), so generation can then run with --offline.

Prints one JSON line per file ({"file", "done", "total"}) so an app can show progress.
"""
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
REPO = "DeepBeepMeep/Wan2.1"
FILES = [
    "umt5-xxl/special_tokens_map.json",
    "umt5-xxl/spiece.model",
    "umt5-xxl/tokenizer.json",
    "umt5-xxl/tokenizer_config.json",
    "Wan2.1_VAE.safetensors",
    "wan2.1_text2video_1.3B_mbf16.safetensors",
    "umt5-xxl/models_t5_umt5-xxl-enc-quanto_int8.safetensors",
]


def main():
    target = ROOT / "ckpts"
    target.mkdir(exist_ok=True)
    for index, name in enumerate(FILES, 1):
        print(json.dumps({"file": name, "done": index - 1, "total": len(FILES)}), flush=True)
        hf_hub_download(REPO, name, local_dir=target)
    print(json.dumps({"file": None, "done": len(FILES), "total": len(FILES)}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
