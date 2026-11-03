# vsflip

**vsflip** is a high-performance native **VapourSynth** plugin and Python wrapper for NVIDIA's [FLIP perceptual metric](https://github.com/NVlabs/flip). It computes frame-by-frame perceptual error maps and scores between a reference and a test video.

## Features

- **Blazing Fast CUDA & CPU Backends**: Accelerated evaluation on NVIDIA GPUs (RTX 2000+) reaching over 115+ FPS on 1080p, with multi-threaded CPU fallback.
- **Fast FP16 Mode (Default)**: Hardware half-precision math offering ~2.4x speedup over FP32 with perceptual difference $\Delta < 2.5 \times 10^{-5}$.
- **Automatic Hardware Residency**: Automatically detects whether input frames reside in CPU RAM or Vulkan GPU VRAM (`std.GPUUpload`) and routes them through direct DMA or Vulkan-CUDA zero-copy interop.
- **Flexible Output**: Default host RAM output for full backward compatibility, with optional Vulkan GPU output (`gpu_output=True`) for all-GPU processing pipelines.
- **Grayscale & Magma Maps**: Output error maps as float grayscale (`GRAYS`) or 3-channel heatmap (`RGBS`).
- **Frame Properties**: Every frame stores its perceptual score in `props.meanFLIPError` (lower is closer; `0.0` is identical).
- **Physical Presets**: Built-in viewing condition presets for monitors, phones, and TVs (`FlipParams`).
- **LDR & HDR**: Supports both standard sRGB and high dynamic range content.

## Performance & Benchmarks

> [!NOTE]
> All benchmarks and throughput tests were performed on an **Ryzen 7 7700X** and an **RTX 3060 (12GB)**.

Benchmarked on 1080p (1920x1080) video:

| Mode / Backend | Precision | Real Video (`BestSource`) | Synthetic Raw Frames | Speedup vs CPU |
| :--- | :---: | :---: | :---: | :---: |
| Multi-threaded CPU (Grayscale) | FP32 | ~2.9 FPS | ~2.2 FPS | 1x (Baseline) |
| Multi-threaded CPU (Magma) | FP32 | ~2.8 FPS | ~2.2 FPS | 1x |
| CUDA FP32 (`fast=False`, Grayscale) | FP32 | ~51 FPS | ~53 FPS | ~18x - 24x |
| CUDA FP32 (`fast=False`, Magma) | FP32 | ~50 FPS | ~50 FPS | ~18x - 23x |
| **CUDA FP16 (`fast=True`, Grayscale)** | **FP16** | **~116 - 120 FPS** | **~175+ FPS** | **~40x - 80x** |
| **CUDA FP16 (`fast=True`, Magma)** | **FP16** | **~112 FPS** | **~150 FPS** | **~39x - 68x** |

> **Recommended Pipeline**: Keeping video frames in host RAM (standard output from decoders like `bs.VideoSource` or `core.ffms2.Source`) is the fastest and most efficient path.

### High-Resolution Scaling (4K & 8K)

Thanks to phase-based GPU workspace buffer aliasing and dynamic multi-threaded concurrency throttling, **vsflip** scales smoothly to ultra-high resolutions without running out of VRAM:

| Resolution | Frame Dimensions | Pixels / Frame | CUDA FP16 Throughput | VRAM / Workspace |
| :--- | :---: | :---: | :---: | :---: |
| **1080p FHD** | 1920 x 1080 | ~2.07 M | **~116 - 120 FPS** (~8.4 ms) | ~91 MB |
| **4K UHD** | 3840 x 2160 | ~8.29 M | **~28 - 33 FPS** (~30 - 35 ms) | ~365 MB |
| **8K UHD** | 7680 x 4320 | ~33.18 M | **~7 FPS** (~140 ms) | ~1.46 GB |

## Requirements

- **Python**: ≥ 3.12
- **VapourSynth**: ≥ R74
- *(Optional for CUDA)*: NVIDIA GPU (RTX 2000 or newer) with display driver installed

## Installation

### 1. Python Package
```bash
pip install vsflip
```

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

# Compare entire videos (returns GRAYS clip, evaluated via CUDA FP16 by default)
result = vsflip_video(ref_rgbs, test_rgbs, params=FlipParams.Monitor.reference)

# Output as Magma color heatmap (returns RGBS clip)
heat_map = vsflip_video(ref_rgbs, test_rgbs, magma=True)

# Strict FP32 precision mode
fp32_result = vsflip_video(ref_rgbs, test_rgbs, fast=False)

# Keep output on GPU (Vulkan frame, requires VapourSynth API 4.3+)
gpu_clip = vsflip_video(ref_rgbs, test_rgbs, gpu_output=True)

# Compare a single frame and return the float mean error
score = vsflip_frame(ref_rgbs, test_rgbs, ref_frame=100, test_frame=100)
print(f"Frame 100 FLIP score: {score:.5f}")

# For HDR videos 
# result = vsflip_video(ref_hdr, test_hdr, dynamic_range="HDR")

result.set_output()
```

## API Reference

### `vsflip_video`
```python
vsflip.vsflip_video(
    ref_clip: vs.VideoNode,
    test_clip: vs.VideoNode,
    dynamic_range: str = "LDR",
    params: Union[_ParamConfig, Dict[str, Any]] = FlipParams.Monitor.reference,
    device: str = "auto",
    cuda_device: int = 0,
    magma: bool = False,
    alignment_to_ref: int = 0,
    fast: bool = True,
    gpu_output: bool = False,
    **kwargs: Any
) -> vs.VideoNode
```
- **`ref_clip`**, **`test_clip`**: Input clips in `vs.RGBS` format. Can reside either on CPU RAM or GPU VRAM (Vulkan); input residency is detected and handled automatically.
- **`fast`**: (default `True`) Enables optimized FP16 CUDA half-precision execution. Set to `False` for full FP32 reference precision.
- **`gpu_output`**: (default `False`) When `False`, returns standard CPU/host memory frames for full backward compatibility. When `True`, returns GPU Vulkan frames (`ffGPUOutput`, requires API 4.3+).
- **`magma`**: (default `False`) If `True`, returns an `RGBS` false-color heatmap; if `False`, returns a `GRAYS` error map.
- **`device`**: `"auto"` (prefers CUDA if an NVIDIA GPU is detected, otherwise CPU), `"cuda"`, or `"cpu"`.


## Parameter Presets (`FlipParams`)

`FlipParams` provides presets modeling human contrast sensitivity for different viewing environments:

| Category | Presets |
| :--- | :--- |
| **Monitor** (50 cm distance) | `FlipParams.Monitor.fhd` (27" 1080p), `mainstream` (27" 1440p), `reference` (31" 4K), `retina` (27" 5K) |
| **Mobile** (17 cm distance) | `FlipParams.Mobile.budget` (6.67" 1080p), `reference` (6.9" 1440p), `retina` (6.9" Retina) |
| **TV** (living room distance) | `FlipParams.TV.reference` (65" 4K at 2.5 m), `retina` (65" 4K at 1.31 m) |

You can also pass custom parameters: `params={"ppd": 67.0}`.

## License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

### Third-Party & Mathematical Integrity
- **NVIDIA FLIP** (`src/include/FLIP.h`): Copyright © 2020-2026 NVIDIA Corporation & Affiliates ([BSD 3-Clause](./LICENSE-THIRD-PARTY)).
  > **Note on Implementation**: The core perceptual metric calculation in **vsflip** is **mathematically identical** to NVIDIA's official reference FLIP implementation (`flip.py` and C++/CUDA reference). All spatial filter convolutions, feature detection matrices, color space transformations (sRGB/linear to YCxCz / Hunter Lab), spatial contrast sensitivity functions (CSF), and error pooling formulas are completely unmodified.
  >
  > The only modifications made to the reference code are:
  > 1. **High-throughput memory management**: Direct planar DMA transfers, CUDA-Vulkan zero-copy driver interop, pre-allocated persistent stream workspaces, and asynchronous frame pipelines designed for VapourSynth.
  > 2. **FP16 half-precision execution (`fast=True`)**: Optimized CUDA half-precision execution for a ~2.4x speedup with negligible numerical difference ($\Delta < 2.5 \times 10^{-5}$ vs FP32). Passing `fast=False` evaluates the exact reference FP32 arithmetic.
  > 3. **Thread-safe initialization**: Lazy initialization of color maps to prevent Windows dynamic library loader-lock deadlocks in multi-threaded host environments.
