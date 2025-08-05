# vsflip

**vsflip** is a Python wrapper for **VapourSynt** that enables perceptual video comparison using the [FLIP metric](https://github.com/NVlabs/flip). It computes frame-by-frame error maps that quantify visual differences between a reference and a test video.

This wrapper integrates the `flip` Python library and works entirely within VapourSynth pipelines.

## 📦 Features

- Frame-by-frame FLIP comparison between two `VideoNode`.
- Outputs GrayscaleS error maps where darker pixels indicate better perceptual similarity.
- Supports both `LDR` and `HDR` evaluation modes.
- Optionally exports error maps as images (`.png`).

## 🛠 Requirements

- Python ≥ 3.13
- [VapourSynth](https://www.vapoursynth.com/) R71 or newer
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [flip](https://github.com/NVlabs/flip)

## 🚀 Installation

You can install `vsflip` via pip:

```bash
pip install vsflip
```

## ✏️ Basic Usage
```python
from vsflip import vsflip_frame, vsflip_video

# Compare the first frame of two clips
result = vsflip_frame(ref_clip, test_clip)

# Compare all frames (must be aligned and same length)
result = vsflip_video(ref_clip, test_clip)
```
- Both functions return a VideoNode (GRAYS format).

- Refer to the Python docstrings for detailed parameter descriptions and options.

## License
This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

### Third-party Dependencies
This project uses the [FLIP metric](https://github.com/NVlabs/flip) python wrapper by NVIDIA, which is licensed under the BSD 3-Clause License.

The flip library's source code or binaries are **not bundled** with this project.
It is installed separately as a required runtime dependency.

## TODO List
- Hoping that Nvidia hasn't dropped the project, I want to create the Python  wrapper and the Vaporsynth implementation for the CUDA version as well. 
  I actually tried, but I miserably failed... Any help is appreciated.
- Fixing Exposure value parsing

## 📬 Contact
For questions, bugs, or feature requests, feel free to open an issue or pull request.