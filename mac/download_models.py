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


# --hd: Wan 2.2 TI2V 5B (int8), its decoder and the FastWan 3-step accelerator (about 8.7 GB more) for the "hd" preset.
HD_FILES = [
    ("loras_accelerators/Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors", "loras/wan_5B"),
    ("Wan2.2_VAE.safetensors", "ckpts"),
    ("wan2.2_text2video_5B_quanto_mbf16_int8.safetensors", "ckpts"),
]
HD_REPO = "DeepBeepMeep/Wan2.2"


def main():
    target = ROOT / "ckpts"
    target.mkdir(exist_ok=True)
    jobs = [(REPO, name, target, None) for name in FILES]
    if "--hd" in sys.argv:
        jobs += [(HD_REPO, name, ROOT / folder, folder) for name, folder in HD_FILES]
    if "--hd-only" in sys.argv:
        jobs = [(HD_REPO, name, ROOT / folder, folder) for name, folder in HD_FILES]
    for index, (repo, name, where, folder) in enumerate(jobs, 1):
        print(json.dumps({"file": name, "done": index - 1, "total": len(jobs)}), flush=True)
        if folder == "loras/wan_5B":  # loras live flat in their folder, not under the repo's subfolder
            where.mkdir(parents=True, exist_ok=True)
            final = where / Path(name).name
            if not final.exists():
                Path(hf_hub_download(repo, name, local_dir=ROOT / ".lora_download")).rename(final)
        else:
            hf_hub_download(repo, name, local_dir=where)
    print(json.dumps({"file": None, "done": len(jobs), "total": len(jobs)}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
