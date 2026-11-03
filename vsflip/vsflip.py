from __future__ import annotations
import os
import sys
from typing import Union, Dict, Any
import vapoursynth as vs

core = vs.core

if not hasattr(core, "vsflip"):
    _candidates = ["vsflip.dll"] if sys.platform == "win32" else ["libvsflip.so", "vsflip.so"]
    for _lib_name in _candidates:
        _lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _lib_name)
        if os.path.isfile(_lib_path):
            core.std.LoadPlugin(_lib_path)
            break

__version__ = "2.1.0"
__all__ = ["__version__", "FlipParams", "vsflip_frame", "vsflip_video"]


class _ParamConfig:
    def __init__(self, category: str, name: str, params_dict: dict):
        self.category = category
        self.name = name
        self.para_dict = params_dict

    def getname(self) -> str:
        return f"{self.category} {self.name}"

    def __repr__(self) -> str:
        return f"<FlipParams: {self.getname()} {self.para_dict}>"


class FlipParams:
    """Preconfigured viewing condition presets for monitors, mobile devices, and TVs."""
    # 50cm distance
    class Monitor:
        hd         = _ParamConfig("monitor", "hd21", {"vc": [0.5, 1280, 0.465], "tonemapper": "ACES"})        # HD 21"
        fhd        = _ParamConfig("monitor", "fhd27", {"vc": [0.5, 1920, 0.598], "tonemapper": "ACES"})       # FHD 27"
        mainstream = _ParamConfig("monitor", "mainstream", {"vc": [0.5, 2560, 0.598], "tonemapper": "ACES"})  # Mainstream (QHD 27")
        reference  = _ParamConfig("monitor", "reference", {"vc": [0.5, 3840, 0.686], "tonemapper": "ACES"})   # Reference (4K 31")
        retina     = _ParamConfig("monitor", "retina", {"vc": [0.5, 5120, 0.598], "tonemapper": "ACES"})      # Retina (5K 27")

    # 17cm distance
    class Mobile:
        budget     = _ParamConfig("mobile", "budget", {"vc": [0.17, 1920, 0.148], "tonemapper": "ACES"})     # Budget (FHD 6.67")
        reference  = _ParamConfig("mobile", "reference", {"vc": [0.17, 2560, 0.153], "tonemapper": "ACES"})  # Reference (QHD 6.9")
        retina     = _ParamConfig("mobile", "retina", {"vc": [0.17, 2293, 0.153], "tonemapper": "ACES"})     # Retina (iPhone Pro Max 6.9")

    # TV viewing distance
    class TV:
        reference = _ParamConfig("tv", "reference", {"vc": [2.5, 3840, 1.439], "tonemapper": "ACES"})         # Reference (4K 65" at 2.5m)
        retina    = _ParamConfig("tv", "retina", {"vc": [1.31, 3840, 1.439], "tonemapper": "ACES"})           # Retina (4K 65" at 1.31m)


def vsflip_video(
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
) -> vs.VideoNode:
    """
    Compare two videos frame-by-frame using the native C++/CUDA FLIP metric.

    Both clips must be in vs.RGBS format (32-bit float planar RGB in linear light).
    Input frames are automatically detected and read directly from GPU (VRAM zero-copy)
    if resident on GPU, or copied from CPU host memory.

    :param ref_clip:          Reference VapourSynth VideoNode (must be vs.RGBS linear).
    :param test_clip:         Test VapourSynth VideoNode (must be vs.RGBS linear).
    :param dynamic_range:     Dynamic range mode: 'LDR' or 'HDR'.
    :param params:            Preset configuration (_ParamConfig) or dictionary of settings.
    :param device:            Computation device: 'cuda', 'cpu', or 'auto' (default: 'auto').
    :param cuda_device:       CUDA GPU device index (default: 0).
    :param magma:             If True, outputs error map as a 3-channel RGB Magma heatmap (RGBS).
                              If False, outputs a 1-channel Grayscale error map (GRAYS).
    :param alignment_to_ref:  Frame offset to align test clip with reference clip.
    :param fast:              If True (default in v2.1), uses FP16 half-precision on CUDA for maximum speed.
    :param gpu_output:        If True, returns a GPU-resident clip (nrGPU via Vulkan).
                              If False (default), returns a standard CPU-resident clip for backward compatibility.
    :return:                  VideoNode with FLIP error map and 'meanFLIPError' frame property.
    """
    if "range" in kwargs:
        dynamic_range = kwargs["range"]
    if "allignment_to_ref" in kwargs:
        alignment_to_ref = kwargs["allignment_to_ref"]
    if "superfast" in kwargs:
        fast = fast or bool(kwargs["superfast"])
    if "gpu_output" in kwargs:
        gpu_output = bool(kwargs["gpu_output"])

    if dynamic_range.upper() not in ("LDR", "HDR"):
        raise ValueError(f"dynamic_range must be 'LDR' or 'HDR', got '{dynamic_range}'.")

    if ref_clip.format is None or ref_clip.format.id != vs.RGBS:
        raise TypeError(
            f"vsflip: 'ref_clip' must be in vs.RGBS format (32-bit float RGB). "
            f"Got {ref_clip.format.name if ref_clip.format else 'unknown'}. "
            "Convert the clip to RGBS beforehand in linear light."
        )

    if test_clip.format is None or test_clip.format.id != vs.RGBS:
        raise TypeError(
            f"vsflip: 'test_clip' must be in vs.RGBS format (32-bit float RGB). "
            f"Got {test_clip.format.name if test_clip.format else 'unknown'}. "
            "Convert the clip to RGBS beforehand in linear light."
        )

    if alignment_to_ref > 0:
        test_clip = test_clip[alignment_to_ref:]
    elif alignment_to_ref < 0:
        ref_clip = ref_clip[-alignment_to_ref:]

    param_dict = params.para_dict if isinstance(params, _ParamConfig) else dict(params)

    dev_str = device.lower().strip()
    if dev_str == "cpu":
        dev_code = 0
    elif dev_str == "cuda":
        dev_code = 1
    else:
        dev_code = 2  # AUTO

    dyn_range_code = 1 if dynamic_range.upper() == "HDR" else 0
    ppd = float(param_dict.get("ppd", 67.0))
    vc = param_dict.get("vc", None)
    tonemapper = str(param_dict.get("tonemapper", "ACES")).lower()
    start_exp = param_dict.get("startExposure", param_dict.get("start_exposure", None))
    stop_exp = param_dict.get("stopExposure", param_dict.get("stop_exposure", None))
    num_exp = param_dict.get("numExposures", param_dict.get("num_exposures", None))

    eval_kwargs: Dict[str, Any] = {
        "reference": ref_clip,
        "test": test_clip,
        "dynamic_range": dyn_range_code,
        "device": dev_code,
        "cuda_device": cuda_device,
        "magma": 1 if magma else 0,
        "tonemapper": tonemapper,
    }

    if vc is not None and len(vc) == 3:
        eval_kwargs["vc"] = [float(v) for v in vc]
    else:
        eval_kwargs["ppd"] = ppd

    if start_exp is not None:
        eval_kwargs["start_exposure"] = float(start_exp)
    if stop_exp is not None:
        eval_kwargs["stop_exposure"] = float(stop_exp)
    if num_exp is not None:
        eval_kwargs["num_exposures"] = int(num_exp)

    eval_kwargs["fast"] = 1 if fast else 0
    eval_kwargs["gpu_output"] = 1 if gpu_output else 0

    node = core.vsflip.Evaluate(**eval_kwargs)
    if not magma:
        node = core.std.RemoveFrameProps(node, ["_Matrix", "_ChromaLocation"])
    return node


def vsflip_frame(
    ref_clip: vs.VideoNode,
    test_clip: vs.VideoNode,
    dynamic_range: str = "LDR",
    ref_frame: int = 0,
    test_frame: int = 0,
    params: Union[_ParamConfig, Dict[str, Any]] = FlipParams.Monitor.reference,
    device: str = "auto",
    cuda_device: int = 0,
    magma: bool = False,
    fast: bool = True,
    gpu_output: bool = False,
    **kwargs: Any
) -> vs.VideoNode:
    """
    Compare two single frames using the native C++/CUDA FLIP metric.

    Both clips must be in vs.RGBS format (32-bit float planar RGB in linear light).
    Input frames are automatically detected and read directly from GPU (VRAM zero-copy)
    if resident on GPU, or copied from CPU host memory.

    :param ref_clip:       Reference VapourSynth VideoNode (must be vs.RGBS linear).
    :param test_clip:      Test VapourSynth VideoNode (must be vs.RGBS linear).
    :param dynamic_range:  Dynamic range mode: 'LDR' or 'HDR'.
    :param ref_frame:      Frame index of the reference clip to compare.
    :param test_frame:     Frame index of the test clip to compare.
    :param params:         Preset configuration (_ParamConfig) or dictionary of settings.
    :param device:         Computation device: 'cuda', 'cpu', or 'auto' (default: 'auto').
    :param cuda_device:    CUDA GPU device index (default: 0).
    :param magma:          If True, outputs error map as a 3-channel RGB Magma heatmap (RGBS).
    :param fast:           If True (default in v2.1), uses FP16 half-precision on CUDA for maximum speed.
    :param gpu_output:     If True, returns a GPU-resident clip (nrGPU via Vulkan).
                           If False (default), returns a standard CPU-resident clip for backward compatibility.
    :return:               1-frame long VideoNode with FLIP error map and 'meanFLIPError' frame property.
    """
    if "range" in kwargs:
        dynamic_range = kwargs["range"]
    if "superfast" in kwargs:
        fast = fast or bool(kwargs["superfast"])
    if "gpu_output" in kwargs:
        gpu_output = bool(kwargs["gpu_output"])

    single_ref = ref_clip[ref_frame:ref_frame + 1]
    single_test = test_clip[test_frame:test_frame + 1]

    return vsflip_video(
        ref_clip=single_ref,
        test_clip=single_test,
        dynamic_range=dynamic_range,
        params=params,
        device=device,
        cuda_device=cuda_device,
        magma=magma,
        fast=fast,
        gpu_output=gpu_output,
    )
