# Mac port of WanGP - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt). Added 2026-10-29.
"""Text-to-video on Apple Silicon with WanGP, as a small Python API.

Presets were measured on an M5 MacBook Pro with 16 GB of memory (Wan 2.1 Text2Video 1.3B):
- "quality": 5 s at 832x480 (81 frames, 16 fps). About 12 minutes.
- "draft":   5 s at 640x352. Faster, softer image.

Speed-ups that keep the image almost unchanged: classifier-free guidance only while the scene is laid out (first phase), and
MagCache, which reuses the model output on steps where it barely changes.
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
    "quality": {"resolution": "832x480"},
    "draft": {"resolution": "640x352"},
}

BASE_SETTINGS = {
    "model_type": MODEL_TYPE,
    "num_inference_steps": 20,
    "guidance_phases": 2,          # guidance while the scene is laid out, then a single pass per step
    "guidance_scale": 5,
    "guidance2_scale": 1,
    "switch_threshold": 800,
    "flow_shift": 5,
    "skip_steps_cache_type": "mag",
    "skip_steps_multiplier": 2.0,
    "skip_steps_start_step_perc": 20,
}


def frames_for(seconds):
    """Wan works with 4n+1 frames; 5 s (81 frames) is the length the model was trained on."""
    frames = int(round(seconds * FPS))
    return max(17, min(81, (frames // 4) * 4 + 1))


class Engine:
    """Loads WanGP once and keeps the model in memory between videos."""

    def __init__(self, output_dir=None, offline=False, log=None):
        self.log = log or (lambda text: None)
        self.output_dir = Path(output_dir).expanduser() if output_dir else None
        args = ["--attention", "sdpa"]
        if offline:
            args.append("--offline")
        from shared.api import init  # imported late: WanGP applies the Apple Silicon patch while it loads
        self.session = init(root=ROOT, cli_args=args)

    def generate(self, prompt, seconds=5, preset="quality", seed=-1, on_progress=None):
        """Generates one video and returns its path. on_progress(phase, step, total, elapsed_seconds)."""
        settings = {**BASE_SETTINGS, **PRESETS.get(preset, PRESETS["quality"]),
                    "prompt": prompt, "video_length": frames_for(seconds), "seed": seed}
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
