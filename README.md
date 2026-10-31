# vsflip

**vsflip** is a high-performance native **VapourSynth** plugin and Python wrapper for NVIDIA's [FLIP perceptual metric](https://github.com/NVlabs/flip). It computes frame-by-frame perceptual error maps and scores between a reference and a test video.

## Features

- **CUDA & CPU Backends**: Accelerated evaluation on NVIDIA GPUs (RTX 2000+) with multi-threaded CPU fallback.
- **Grayscale & Magma Maps**: Output error maps as float grayscale (`GRAYS`) or 3-channel heatmap (`RGBS`).
- **Frame Properties**: Every frame stores its perceptual score in `props.meanFLIPError` (lower is closer; `0.0` is identical).
- **Physical Presets**: Built-in viewing condition presets for monitors, phones, and TVs (`FlipParams`).
- **LDR & HDR**: Supports both standard sRGB and high dynamic range content.

## Requirements

- **Python**: ≥ 3.12
- **VapourSynth**: ≥ R74
- *(Optional for CUDA)*: NVIDIA GPU (RTX 2000 or newer) with display driver installed

## Installation

### 1. Python Package
```bash
pip install vsflip
```
*(Pre-compiled wheels bundle the native binary out of the box).*

### 2. Manual Compilation (Optional)
If building from source:
```powershell
cmake -B build -A x64 -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;89-virtual"
cmake --build build --config Release
```
Place `vsflip.dll` (Windows) or `libvsflip.so` (Linux) in your VapourSynth `plugins/` directory or inside the `vsflip/` folder.

## Usage

```python
import vapoursynth as vs
core = vs.core
from vsflip import vsflip_video, vsflip_frame, FlipParams

ref = core.ffms2.Source("reference.mkv")
test = core.ffms2.Source("encoded.mkv")

# Convert clips to vs.RGBS with the appropriate matrix/transfer for your source
ref_rgbs = core.resize.Bicubic(ref, format=vs.RGBS, matrix_in=1)
test_rgbs = core.resize.Bicubic(test, format=vs.RGBS, matrix_in=1)

# Compare entire videos (returns GRAYS clip)
result = vsflip_video(ref_rgbs, test_rgbs, params=FlipParams.Monitor.reference, device="cuda")

# Output as Magma color heatmap (returns RGBS clip)
heat_map = vsflip_video(ref_rgbs, test_rgbs, magma=True, device="cuda")

# Compare a single frame
single = vsflip_frame(ref_rgbs, test_rgbs, ref_frame=100, test_frame=100)

# For HDR videos (e.g. BT.2020 PQ), convert to Linear RGBS (transfer='linear'):
# ref_hdr = core.resize.Bicubic(ref, format=vs.RGBS, matrix_in=9, transfer_in=16, transfer="linear")
# test_hdr = core.resize.Bicubic(test, format=vs.RGBS, matrix_in=9, transfer_in=16, transfer="linear")
# result = vsflip_video(ref_hdr, test_hdr, dynamic_range="HDR")

result.set_output()
```

## Parameter Presets (`FlipParams`)

`FlipParams` provides presets modeling human contrast sensitivity for different viewing environments:

| Category | Presets |
| :--- | :--- |
| **Monitor** (50 cm distance) | `FlipParams.Monitor.fhd` (27" 1080p), `mainstream` (27" 1440p), `reference` (31" 4K), `retina` (27" 5K) |
| **Mobile** (17 cm distance) | `FlipParams.Mobile.budget` (6.67" 1080p), `reference` (6.9" 1440p), `retina` (6.9" Retina) |
| **TV** (living room distance) | `FlipParams.TV.reference` (65" 4K at 2.5 m), `retina` (65" 4K at 1.31 m) |

You can also pass a custom dict: `params={"ppd": 67.0}`.

## License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

### Third-Party
- **NVIDIA FLIP** (`src/include/FLIP.h`): Copyright © 2020-2026 NVIDIA Corporation & Affiliates ([BSD 3-Clause](./LICENSE-THIRD-PARTY)).
  > **Note**: `src/include/FLIP.h` contains minor adjustments (thread-safe lazy initialization of color maps) to prevent Windows dynamic library loader-lock deadlocks, ensuring seamless compatibility within the VapourSynth plugin environment while preserving all perceptual algorithms and formulas intact.
