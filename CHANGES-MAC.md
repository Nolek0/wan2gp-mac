# Changes in this Mac port

This repository is a modified version of [WanGP](https://github.com/deepbeepmeep/Wan2GP) by DeepBeepMeep, redistributed for free use under the WanGP Community License 2.0 (see `LICENSE.txt`). It is an unofficial port: it is not made, reviewed or endorsed by DeepBeepMeep.

Changes made on **2026-10-29** by Nolek0:

| File | Change |
|---|---|
| `shared/mps/device_patch.py` | CUDA events and streams emulated on Apple Silicon now wait for the GPU (`torch.mps.synchronize`) instead of returning at once. With the old no-op events, mmgp reused host buffers that the GPU had not read yet, which overwrote the latents from the second denoising step on: every video came out as noise or black. The emulated device properties also gain `L2_cache_size`, `shared_memory_per_block_optin` and `gcnArchName`, which WanGP's kernels read. |
| `models/wan/any2video.py` | On Apple Silicon the umT5 text encoder (6 GB) is loaded only to encode a new prompt, run block by block on the GPU (about 4 s) and released right after (`OnDemandT5`): kept resident, it took shared memory away from the denoising for the whole video. |
| `models/wan/wan_handler.py` | Plain text-to-video no longer downloads or requires the CLIP image encoder and the image-only upscaling VAE (2.9 GB). The on-demand text encoder is not handed to mmgp. |
| `mmgp/pinned.py` | No page locking on macOS (`cudaHostRegister` does not exist there, and CPU and GPU already share the same memory). The buffers are still budgeted and released the same way. |
| `mac/` (new) | `setup.sh` (environment and models), `download_models.py` (only the ~10 GB the 1.3B text-to-video model needs), `engine.py` (tested presets, 256-pixel decoding tiles), `generate.py` (command line) and `serve.py` (JSON-lines service for other apps). Peak memory 3.7 GB instead of 9.1 GB. |
| `shared/mps/device_patch.py` (2) | Each `scaled_dot_product_attention` call on MPS ends with a GPU synchronization, and `mac/engine.py` splits attention heads into 16 groups. A 720p video attention queued as one long GPU job starved macOS long enough for the kernel watchdog to restart the Mac; with short jobs the system never stalled more than a few seconds. |
| `models/wan/modules/vae2_2.py` | `WANGP_VAE22_TILE` sets the Wan 2.2 decoder tile size (the presets only offer 256 pixels or whole frames, which ran out of GPU memory on a 16 GB Mac). |
| `mac/engine.py` (2) | `hd` preset: Wan 2.2 TI2V 5B (int8) with the FastWan 3-step accelerator, 960x544 at 24 fps; a negative prompt against smeared or see-through motion for the guided presets. `download_models.py --hd` fetches its files (about 8.7 GB). |
| `README.md` | A Mac section at the top. The original README follows it unchanged. |

Everything else is WanGP as published upstream at commit `97845dc`.
