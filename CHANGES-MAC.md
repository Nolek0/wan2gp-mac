# Changes in this Mac port

This repository is a modified version of [WanGP](https://github.com/deepbeepmeep/Wan2GP) by DeepBeepMeep, redistributed for free use under the WanGP Community License 2.0 (see `LICENSE.txt`). It is an unofficial port: it is not made, reviewed or endorsed by DeepBeepMeep.

Changes made on **2026-10-29** by Nolek0:

| File | Change |
|---|---|
| `shared/mps/device_patch.py` | CUDA events and streams emulated on Apple Silicon now wait for the GPU (`torch.mps.synchronize`) instead of returning at once. With the old no-op events, mmgp reused host buffers that the GPU had not read yet, which overwrote the latents from the second denoising step on: every video came out as noise or black. The emulated device properties also gain `L2_cache_size`, `shared_memory_per_block_optin` and `gcnArchName`, which WanGP's kernels read. |
| `mmgp/pinned.py` | No page locking on macOS (`cudaHostRegister` does not exist there, and CPU and GPU already share the same memory). The buffers are still budgeted and released the same way. |
| `mac/` (new) | `setup.sh` (environment and models), `download_models.py` (only the ~10 GB the 1.3B text-to-video model needs), `engine.py` (tested presets), `generate.py` (command line) and `serve.py` (JSON-lines service for other apps). |
| `README.md` | A Mac section at the top. The original README follows it unchanged. |

Everything else is WanGP as published upstream at commit `97845dc`.
