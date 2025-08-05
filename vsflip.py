try:
    import vapoursynth as vs
except ImportError:
    raise ImportError('Vapoursynth R71> is required')

import flip_evaluator as flip
from matplotlib.pyplot import imsave
import numpy as np
from weakref import WeakValueDictionary

core = vs.core

#What this value means:
#First float is the distance to display (meters), second is display width (pixels), and third is display width (meters).
#Tonemapper allowed options are "ACES", "Hable", and "Reinhard"
#Exposure values should be left as None, let FLIP calculate them. This doesn't apply if you're doing an HDR comparison."

#TODO:
#Fixing Exposure value parsing
FLIPDICT = {
        "vc": [0.5, 3840, 0.6],
        "tonemapper": "ACES",
    }

_f2c_cache = WeakValueDictionary[int, vs.VideoNode]()

def frame2clip(frame: vs.VideoFrame) -> vs.VideoNode:
    """
    Original code from [vsjetpack](https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack/blob/main/vstools/functions/utils.py#L369)

    Convert a VideoFrame to a VideoNode.

    :param frame:       Input frame.

    :return:            1-frame long VideoNode of the input frame.
    """

    key = hash((frame.width, frame.height, frame.format.id))

    if _f2c_cache.get(key, None) is None:
        _f2c_cache[key] = blank_clip = vs.core.std.BlankClip(
            None, frame.width, frame.height,
            frame.format.id, 1, 1, 1,
            [0] * frame.format.num_planes,
            True
        )
    else:
        blank_clip = _f2c_cache[key]

    frame_cp = frame.copy()

    return vs.core.std.ModifyFrame(blank_clip, blank_clip, lambda n, f: frame_cp)


def frame_to_numpyArray(frame: vs.VideoNode) -> np.ndarray:
    """
    Convert a VapourSynth 1-frame long VideoNode to a NumPy array.
    
    :param frame: The VapourSynth 1-frame long VideoNode to convert.
    :return: A NumPy array representation of the frame.
    """
    frame = frame.get_frame(0)

    return  np.stack([np.asarray(frame[i]) for i in range(frame.format.num_planes)], axis=-1)


def numpy_to_frame(np_array: np.ndarray, blank_clip1 : vs.VideoNode) -> vs.VideoFrame:
    """
    Convert a 2D NumPy array (float32) to VapourSynth VideoFrame in GRAYS.
    """
    assert np_array.ndim == 2 and np_array.dtype == np.float32, "Input must be a 2D NumPy array of type float32."
    
    if blank_clip1 is None:
        height, width = np_array.shape
        blank_clip1 = core.std.BlankClip(format=vs.GRAYS, width=width, height=height, length=1) 

    frame = blank_clip1.get_frame(0).copy()

    np.copyto(np.asarray(frame[0]), np_array)

    return frame


def vsflip_frame (
        ref_clip:vs.VideoNode,
        test_clip:vs.VideoNode,
        range:str="LDR",
        ref_frame: int = 0,
        test_frame: int = 0,
        parameters: dict = FLIPDICT,
        save_flip_error_mask: bool = False,
        debug: bool = False,
        blank: vs.VideoNode = None
)-> vs.VideoNode:
    """
    Compare two frames using the FLIP metric. Darker values indicate a better match.

    :param ref_clip:                The reference VapourSynth VideoNode.
    :param test_clip:               The test VapourSynth VideoNode.   
    :param range:                   The range of the video, either "LDR" or "HDR".
    :param ref_frame:               The frame number of the reference clip to compare. Default is 0.
    :param test_frame:              The frame number of the test clip to compare. Default is 0.
    :param parameters:              A dictionary of parameters for the FLIP evaluation. Default is {"vc": [0.5, 3840, 0.6], "tonemapper": "ACES"}).
    :param save_flip_error_mask:    If True, saves the FLIP error mask as png in the script folder. Default is False.
    :param debug:                   If True, prints debug information. Default is False.
    :param blank:                   Only for devoloper use.
    :return:                        A VapourSynth 1-frame long VideoNode containing the FLIP error map in GrayScaleS.
    """
    
    if range not in ["LDR", "HDR"]:
        raise ValueError("Range must be either 'LDR' or 'HDR'.")
    
    frame_test=test_clip[test_frame]
    frame_ref=ref_clip[ref_frame]

    if debug:
        print("Reference Frame Properties:\n")
        print(frame_ref)
        print("\nTest Frame Properties:\n")
        print(frame_test)

    try:
        if frame_ref.format.id != vs.RGBS or frame_test.format.id != vs.RGBS:
            frame_ref= core.resize.Bicubic(frame_ref, format=vs.RGBS)
            frame_test = core.resize.Bicubic(frame_test, format=vs.RGBS)

        np_ref = frame_to_numpyArray(frame_ref)
        np_test = frame_to_numpyArray(frame_test)

        flipErrorMap, meanFLIPError, parameters = flip.evaluate(np_ref, np_test, range, applyMagma=False, parameters=parameters)
        flipErrorMap = np.squeeze(flipErrorMap)

        if debug:
            print("Mean FLIP error: ", round(meanFLIPError, 6), "\n")

            print("The following parameters were used:")
            for key in parameters:
                val = parameters[key]
                if isinstance(val, float):
                    val = round(val, 4)
                print("\t%s: %s" % (key, str(val)))

        if save_flip_error_mask:
            imsave(f"vsflip_error_map_{ref_frame}_{test_frame}.png", flipErrorMap, cmap="gray")

        if debug:
            print("numpy array max and min values (correct value should be between 1-0):")
            print(flipErrorMap.max(), flipErrorMap.min())

        frame= numpy_to_frame(flipErrorMap, blank_clip1=blank)
        
        # Add meanFLIPError to frame properties
        frame.props["meanFLIPError"] = float(meanFLIPError)
        
        return frame2clip(frame)

    except Exception as e:
        print(f"Error processing frame {ref_frame} and {test_frame}: {e}")
        return blank # Return a blank frame in case of error


def vsflip_video(
        ref_clip: vs.VideoNode,
        test_clip: vs.VideoNode,
        range: str = "LDR",
        parameters: dict = FLIPDICT,
        debug: bool = False,
        allignment_to_ref: int = 0
) -> vs.VideoNode:
    """
    Compare two videos using the FLIP metric frame by frame. Darker values indicate a better match.
    Both clips must have the same number of frames or at least be perfectly aligned.
    
    :param ref_clip:                The reference VapourSynth VideoNode.
    :param test_clip:               The test VapourSynth VideoNode.
    :param range:                   The range of the video, either "LDR" or "HDR".
    :param parameters:              A dictionary of parameters for the FLIP evaluation. Default is {"vc": [0.5, 3840, 0.6], "tonemapper": "ACES"}).
    :param debug:                   If True, prints debug information. Default is False.
    :param allignment_to_ref:       The number of frames to align the test clip to the reference clip. Default is 0.
                                    Ensure both clip are perfectly aligned, because using this offset could lead to error. 
    :return:                        A VapourSynth VideoNode containing the FLIP error map for each frame in GrayScaleS.
    """
    
    # just in case the user is tard
    min_length = min(ref_clip.num_frames, test_clip.num_frames - allignment_to_ref)

    blank = core.std.BlankClip(
        format=vs.GRAYS,
        width=ref_clip.width,
        height=ref_clip.height,
        length=min_length,
        fpsnum=ref_clip.fps_num,
        fpsden=ref_clip.fps_den
    ) 

    blank1 = core.std.BlankClip(
        format=vs.GRAYS,
        width=ref_clip.width,
        height=ref_clip.height,
        length=1,
    ) 

    def select_flip(n: int) -> vs.VideoNode:
        try:
            return vsflip_frame(
                ref_clip,
                test_clip,
                range=range,
                ref_frame=n,
                test_frame=n+allignment_to_ref,
                parameters=parameters,
                save_flip_error_mask=False,
                debug=debug,
                blank=blank1
            )
        except Exception as e:
            print(f"Error processing frame {n}: {e}")
            return blank1 # Return a blank frame in case of error

    return core.std.FrameEval(clip=blank, eval=select_flip)

