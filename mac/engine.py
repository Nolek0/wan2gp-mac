# Mac port of WanGP - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt). Added 2026-10-29.
"""Text-to-video on Apple Silicon with WanGP, as a small Python API.

Presets were measured on an M5 MacBook Pro with 16 GB of memory (Wan 2.1 Text2Video 1.3B), peak memory about 3.7 GB:
- "quality": 5 s at 720x416 (81 frames, 16 fps). About 7 minutes.
- "high":    5 s at 832x480. About 12 minutes.
- "draft":   5 s at 640x352. Faster, softer image.
- "hd":      5 s at 960x544, 24 fps, with Wan 2.2 TI2V 5B + FastWan (3 steps). About 9 minutes, peak memory about 7 GB, much
             sharper people and faces. Needs the extra files from `download_models.py --hd`.

Speed-ups that keep the image almost unchanged: classifier-free guidance only while the scene is laid out (first phase), and
MagCache, which reuses the model output on steps where it barely changes. Most of each step is attention, which grows with the
square of the pixel count, so 720x416 costs about half of 832x480.
"""
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_TYPE = "t2v_1.3B"
FPS = 16

PRESETS = {
    "quality": {"resolution": "720x416"},
    "high": {"resolution": "832x480"},
    "draft": {"resolution": "640x352"},
    # Wan 2.2 5B distilled to 3 steps by FastWan (no guidance). Most of the time is the Wan 2.2 decoder, which is compute bound.
    "hd": {"model_type": "ti2v_2_2_fastwan", "resolution": "960x544", "num_inference_steps": 3, "guidance_phases": 1,
           "guidance_scale": 1, "flow_shift": 3, "skip_steps_cache_type": ""},
}
FPS_BY_PRESET = {"hd": 24}

BASE_SETTINGS = {
    "model_type": MODEL_TYPE,
    "num_inference_steps": 20,
    "guidance_phases": 2,          # guidance while the scene is laid out, then a single pass per step
    "guidance_scale": 5,
    "guidance2_scale": 1,
    "switch_threshold": 850,
    "flow_shift": 5,
    "skip_steps_cache_type": "mag",
    "skip_steps_multiplier": 2.25,
    "skip_steps_start_step_perc": 10,
}


# Steers the model away from the defects of fast motion (smeared, see-through bodies) and toward a natural, sharp image.
NEGATIVE_PROMPT = ("blurry, motion blur, smeared, ghosting, transparent body, deformed, distorted face, extra limbs, bad hands, "
                   "low quality, jpeg artifacts, overexposed, static, cartoon")


def frames_for(seconds, fps=FPS):
    """Wan works with 4n+1 frames; 5 s is the length the models were trained on (81 frames at 16 fps, 121 at 24 fps)."""
    frames = int(round(seconds * fps))
    return max(17, min(5 * fps + 1, (frames // 4) * 4 + 1))


class Engine:
    """Loads WanGP once and keeps the model in memory between videos."""

    def __init__(self, output_dir=None, offline=False, log=None):
        self.log = log or (lambda text: None)
        self.output_dir = Path(output_dir).expanduser() if output_dir else None
        # --vae-config 2: 256-pixel decoding tiles. WanGP sizes them from the GPU memory, which on a Mac is the whole RAM, and the
        # larger tiles more than doubled the peak memory (8.6 GB instead of 3.4 GB) for the same decoding time.
        args = ["--attention", "sdpa", "--vae-config", "2"]
        if offline:
            args.append("--offline")
        from shared.api import init  # imported late: WanGP applies the Apple Silicon patch while it loads
        self.session = init(root=ROOT, cli_args=args)
        if sys.platform == "darwin":
            # Attention in 16 head groups, each its own GPU job: one 720p attention queued in a single job starved macOS of GPU time
            # long enough for the kernel watchdog to restart the Mac.
            from shared import attention_kit
            attention_kit.configure(3)

    def generate(self, prompt, seconds=5, preset="quality", seed=-1, on_progress=None, negative_prompt=NEGATIVE_PROMPT):
        """Generates one video and returns its path. on_progress(phase, step, total, elapsed_seconds)."""
        settings = {**BASE_SETTINGS, **PRESETS.get(preset, PRESETS["quality"]),
                    "prompt": prompt, "video_length": frames_for(seconds, FPS_BY_PRESET.get(preset, FPS)), "seed": seed}
        if negative_prompt and settings.get("guidance_scale", 1) > 1:
            settings["negative_prompt"] = negative_prompt
        start = time.time()
        job = self.session.submit_task(settings)
        last = None
        for event in job.events.iter(timeout=0.5):
            if event.kind == "progress" and on_progress is not None:
                p = event.data
                key = (p.phase, p.current_step)
                if key != last and p.current_step is not None:
                    last = key
                    on_progress(p.phase, p.current_step, p.total_steps, time.time() - start)
            elif event.kind == "stream":
                text = event.data.text
                if text.strip() and "%|" not in text and "it/s" not in text and "B/s" not in text:
                    self.log(text[:300])
        result = job.result()
        if not result.success:
            raise RuntimeError("; ".join(e.message for e in result.errors) or "generation failed")
        path = Path(result.generated_files[-1])
        if self.output_dir is not None:  # WanGP's --output-dir only applies to its CLI queue mode
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = Path(shutil.move(str(path), self.output_dir / path.name))
        return str(path)
