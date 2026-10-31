#include "include/VapourSynth4.h"
#include "include/VSHelper4.h"
#include "flip_backend.h"

#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <limits>
#include <algorithm>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

static inline float srgb_to_linear(float c) noexcept {
    if (c <= 0.04045f) {
        return (c <= 0.0f) ? 0.0f : (c / 12.92f);
    }
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

struct FlipFilterData {
    VSNode* ref_node;
    VSNode* test_node;
    VSVideoInfo vi;
    FlipEvaluationParams params;
    FlipDevice device;
    int cuda_device_id;
};

static const VSFrame* VS_CC flipGetFrame(
    int n,
    int activationReason,
    void* instanceData,
    void** frameData,
    VSFrameContext* frameCtx,
    VSCore* core,
    const VSAPI* vsapi
) {
    auto* d = static_cast<FlipFilterData*>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->ref_node, frameCtx);
        vsapi->requestFrameFilter(n, d->test_node, frameCtx);
        return nullptr;
    }

    if (activationReason == arAllFramesReady) {
        const VSFrame* ref_frame = vsapi->getFrameFilter(n, d->ref_node, frameCtx);
        const VSFrame* test_frame = vsapi->getFrameFilter(n, d->test_node, frameCtx);

        if (!ref_frame || !test_frame) {
            vsapi->setFilterError("vsflip: failed to retrieve input frames.", frameCtx);
            if (ref_frame) vsapi->freeFrame(ref_frame);
            if (test_frame) vsapi->freeFrame(test_frame);
            return nullptr;
        }

        int width = d->vi.width;
        int height = d->vi.height;
        size_t num_pixels = static_cast<size_t>(width) * height;

        std::vector<float> ref_interleaved(num_pixels * 3);
        std::vector<float> test_interleaved(num_pixels * 3);

        ptrdiff_t ref_stride_r = vsapi->getStride(ref_frame, 0);
        ptrdiff_t ref_stride_g = vsapi->getStride(ref_frame, 1);
        ptrdiff_t ref_stride_b = vsapi->getStride(ref_frame, 2);

        ptrdiff_t test_stride_r = vsapi->getStride(test_frame, 0);
        ptrdiff_t test_stride_g = vsapi->getStride(test_frame, 1);
        ptrdiff_t test_stride_b = vsapi->getStride(test_frame, 2);

        const auto* ref_r = reinterpret_cast<const uint8_t*>(vsapi->getReadPtr(ref_frame, 0));
        const auto* ref_g = reinterpret_cast<const uint8_t*>(vsapi->getReadPtr(ref_frame, 1));
        const auto* ref_b = reinterpret_cast<const uint8_t*>(vsapi->getReadPtr(ref_frame, 2));

        const auto* test_r = reinterpret_cast<const uint8_t*>(vsapi->getReadPtr(test_frame, 0));
        const auto* test_g = reinterpret_cast<const uint8_t*>(vsapi->getReadPtr(test_frame, 1));
        const auto* test_b = reinterpret_cast<const uint8_t*>(vsapi->getReadPtr(test_frame, 2));

        bool should_linearize = (d->params.input_srgb && d->params.dynamic_range == 0);

        if (should_linearize) {
            for (int y = 0; y < height; ++y) {
                const auto* rr = reinterpret_cast<const float*>(ref_r + y * ref_stride_r);
                const auto* rg = reinterpret_cast<const float*>(ref_g + y * ref_stride_g);
                const auto* rb = reinterpret_cast<const float*>(ref_b + y * ref_stride_b);

                const auto* tr = reinterpret_cast<const float*>(test_r + y * test_stride_r);
                const auto* tg = reinterpret_cast<const float*>(test_g + y * test_stride_g);
                const auto* tb = reinterpret_cast<const float*>(test_b + y * test_stride_b);

                float* ref_row = ref_interleaved.data() + static_cast<size_t>(y) * width * 3;
                float* test_row = test_interleaved.data() + static_cast<size_t>(y) * width * 3;

                for (int x = 0; x < width; ++x) {
                    ref_row[x * 3 + 0] = srgb_to_linear(rr[x]);
                    ref_row[x * 3 + 1] = srgb_to_linear(rg[x]);
                    ref_row[x * 3 + 2] = srgb_to_linear(rb[x]);

                    test_row[x * 3 + 0] = srgb_to_linear(tr[x]);
                    test_row[x * 3 + 1] = srgb_to_linear(tg[x]);
                    test_row[x * 3 + 2] = srgb_to_linear(tb[x]);
                }
            }
        } else {
            for (int y = 0; y < height; ++y) {
                const auto* rr = reinterpret_cast<const float*>(ref_r + y * ref_stride_r);
                const auto* rg = reinterpret_cast<const float*>(ref_g + y * ref_stride_g);
                const auto* rb = reinterpret_cast<const float*>(ref_b + y * ref_stride_b);

                const auto* tr = reinterpret_cast<const float*>(test_r + y * test_stride_r);
                const auto* tg = reinterpret_cast<const float*>(test_g + y * test_stride_g);
                const auto* tb = reinterpret_cast<const float*>(test_b + y * test_stride_b);

                float* ref_row = ref_interleaved.data() + static_cast<size_t>(y) * width * 3;
                float* test_row = test_interleaved.data() + static_cast<size_t>(y) * width * 3;

                for (int x = 0; x < width; ++x) {
                    ref_row[x * 3 + 0] = rr[x];
                    ref_row[x * 3 + 1] = rg[x];
                    ref_row[x * 3 + 2] = rb[x];

                    test_row[x * 3 + 0] = tr[x];
                    test_row[x * 3 + 1] = tg[x];
                    test_row[x * 3 + 2] = tb[x];
                }
            }
        }

        size_t out_floats = d->params.apply_magma ? (num_pixels * 3) : num_pixels;
        std::vector<float> output_data(out_floats, 0.0f);
        double mean_error = -1.0;
        bool ok = false;

        bool use_cuda = (d->device == FlipDevice::CUDA) ||
                        (d->device == FlipDevice::AUTO && flip_cuda_is_available());

        if (use_cuda) {
            ok = flip_evaluate_cuda(
                ref_interleaved.data(),
                test_interleaved.data(),
                width,
                height,
                d->params,
                d->cuda_device_id,
                output_data.data(),
                mean_error
            );
        }

        // Fallback to CPU if CUDA failed or was not requested
        if (!ok) {
            ok = flip_evaluate_cpu(
                ref_interleaved.data(),
                test_interleaved.data(),
                width,
                height,
                d->params,
                output_data.data(),
                mean_error
            );
        }

        if (!ok) {
            vsapi->setFilterError("vsflip: metric evaluation failed.", frameCtx);
            vsapi->freeFrame(ref_frame);
            vsapi->freeFrame(test_frame);
            return nullptr;
        }

        VSFrame* out_frame = vsapi->newVideoFrame(&d->vi.format, width, height, ref_frame, core);
        if (!out_frame) {
            vsapi->setFilterError("vsflip: failed to allocate output frame.", frameCtx);
            vsapi->freeFrame(ref_frame);
            vsapi->freeFrame(test_frame);
            return nullptr;
        }

        if (d->params.apply_magma) {
            ptrdiff_t out_stride_r = vsapi->getStride(out_frame, 0);
            ptrdiff_t out_stride_g = vsapi->getStride(out_frame, 1);
            ptrdiff_t out_stride_b = vsapi->getStride(out_frame, 2);

            auto* out_r = reinterpret_cast<uint8_t*>(vsapi->getWritePtr(out_frame, 0));
            auto* out_g = reinterpret_cast<uint8_t*>(vsapi->getWritePtr(out_frame, 1));
            auto* out_b = reinterpret_cast<uint8_t*>(vsapi->getWritePtr(out_frame, 2));

            for (int y = 0; y < height; ++y) {
                auto* dr = reinterpret_cast<float*>(out_r + y * out_stride_r);
                auto* dg = reinterpret_cast<float*>(out_g + y * out_stride_g);
                auto* db = reinterpret_cast<float*>(out_b + y * out_stride_b);
                const float* src_row = output_data.data() + static_cast<size_t>(y) * width * 3;

                for (int x = 0; x < width; ++x) {
                    dr[x] = src_row[x * 3 + 0];
                    dg[x] = src_row[x * 3 + 1];
                    db[x] = src_row[x * 3 + 2];
                }
            }
        } else {
            ptrdiff_t out_stride = vsapi->getStride(out_frame, 0);
            auto* out_ptr = reinterpret_cast<uint8_t*>(vsapi->getWritePtr(out_frame, 0));

            for (int y = 0; y < height; ++y) {
                auto* dst_row = reinterpret_cast<float*>(out_ptr + y * out_stride);
                const float* src_row = output_data.data() + static_cast<size_t>(y) * width;
                std::memcpy(dst_row, src_row, width * sizeof(float));
            }
        }

        VSMap* props = vsapi->getFramePropertiesRW(out_frame);
        vsapi->mapSetFloat(props, "meanFLIPError", mean_error, maReplace);
        vsapi->mapSetInt(props, "_ColorRange", 0, maReplace);
        if (!d->params.apply_magma) {
            vsapi->mapDeleteKey(props, "_Matrix");
            vsapi->mapDeleteKey(props, "_ChromaLocation");
        }

        vsapi->freeFrame(ref_frame);
        vsapi->freeFrame(test_frame);

        return out_frame;
    }

    return nullptr;
}

static void VS_CC flipFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<FlipFilterData*>(instanceData);
    if (d) {
        if (d->ref_node) vsapi->freeNode(d->ref_node);
        if (d->test_node) vsapi->freeNode(d->test_node);
        delete d;
    }
}

static void VS_CC flipCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    int err = 0;

    VSNode* ref_node = vsapi->mapGetNode(in, "reference", 0, &err);
    if (err) {
        vsapi->mapSetError(out, "vsflip: 'reference' clip is required.");
        return;
    }

    VSNode* test_node = vsapi->mapGetNode(in, "test", 0, &err);
    if (err) {
        vsapi->freeNode(ref_node);
        vsapi->mapSetError(out, "vsflip: 'test' clip is required.");
        return;
    }

    const VSVideoInfo* ref_vi = vsapi->getVideoInfo(ref_node);
    const VSVideoInfo* test_vi = vsapi->getVideoInfo(test_node);

    bool is_rgbs_ref = (ref_vi->format.colorFamily == cfRGB && ref_vi->format.sampleType == stFloat && ref_vi->format.bitsPerSample == 32);
    bool is_rgbs_test = (test_vi->format.colorFamily == cfRGB && test_vi->format.sampleType == stFloat && test_vi->format.bitsPerSample == 32);

    if (!is_rgbs_ref || !is_rgbs_test) {
        vsapi->freeNode(ref_node);
        vsapi->freeNode(test_node);
        vsapi->mapSetError(out, "vsflip: both input clips must be in RGBS format.");
        return;
    }

    if (ref_vi->width != test_vi->width || ref_vi->height != test_vi->height) {
        vsapi->freeNode(ref_node);
        vsapi->freeNode(test_node);
        vsapi->mapSetError(out, "vsflip: reference and test clips must have identical width and height.");
        return;
    }

    auto* d = new FlipFilterData();
    d->ref_node = ref_node;
    d->test_node = test_node;
    d->vi = *ref_vi;
    d->vi.numFrames = std::min(ref_vi->numFrames, test_vi->numFrames);

    // Parameters
    d->params.dynamic_range = static_cast<int>(vsapi->mapGetInt(in, "dynamic_range", 0, &err));
    if (err) d->params.dynamic_range = 0; // default LDR

    d->params.ppd = static_cast<float>(vsapi->mapGetFloat(in, "ppd", 0, &err));
    if (err) d->params.ppd = 67.0f;

    int vc_count = vsapi->mapNumElements(in, "vc");
    if (vc_count == 3) {
        d->params.vc[0] = static_cast<float>(vsapi->mapGetFloat(in, "vc", 0, nullptr));
        d->params.vc[1] = static_cast<float>(vsapi->mapGetFloat(in, "vc", 1, nullptr));
        d->params.vc[2] = static_cast<float>(vsapi->mapGetFloat(in, "vc", 2, nullptr));
        d->params.use_vc = true;
    } else {
        d->params.use_vc = false;
        d->params.vc[0] = d->params.vc[1] = d->params.vc[2] = 0.0f;
    }

    const char* tonemapper_str = vsapi->mapGetData(in, "tonemapper", 0, &err);
    if (!err && tonemapper_str) {
        d->params.tonemapper = tonemapper_str;
        std::transform(d->params.tonemapper.begin(), d->params.tonemapper.end(), d->params.tonemapper.begin(), ::tolower);
    } else {
        d->params.tonemapper = "aces";
    }

    d->params.start_exposure = static_cast<float>(vsapi->mapGetFloat(in, "start_exposure", 0, &err));
    if (err) d->params.start_exposure = std::numeric_limits<float>::infinity();

    d->params.stop_exposure = static_cast<float>(vsapi->mapGetFloat(in, "stop_exposure", 0, &err));
    if (err) d->params.stop_exposure = std::numeric_limits<float>::infinity();

    d->params.num_exposures = static_cast<int>(vsapi->mapGetInt(in, "num_exposures", 0, &err));
    if (err) d->params.num_exposures = -1;

    int magma_opt = static_cast<int>(vsapi->mapGetInt(in, "magma", 0, &err));
    d->params.apply_magma = (!err && magma_opt != 0);

    int srgb_opt = static_cast<int>(vsapi->mapGetInt(in, "input_srgb", 0, &err));
    d->params.input_srgb = (err || srgb_opt != 0); // default true

    int dev_opt = static_cast<int>(vsapi->mapGetInt(in, "device", 0, &err));
    if (err || dev_opt == 2) {
        d->device = FlipDevice::AUTO;
    } else if (dev_opt == 1) {
        d->device = FlipDevice::CUDA;
    } else {
        d->device = FlipDevice::CPU;
    }

    d->cuda_device_id = static_cast<int>(vsapi->mapGetInt(in, "cuda_device", 0, &err));
    if (err) d->cuda_device_id = 0;

    if (d->params.apply_magma) {
        vsapi->queryVideoFormat(&d->vi.format, cfRGB, stFloat, 32, 0, 0, core);
    } else {
        vsapi->queryVideoFormat(&d->vi.format, cfGray, stFloat, 32, 0, 0, core);
    }

    VSFilterDependency deps[] = {
        { d->ref_node, rpStrictSpatial },
        { d->test_node, rpStrictSpatial }
    };

    vsapi->createVideoFilter(
        out,
        "Evaluate",
        &d->vi,
        flipGetFrame,
        flipFree,
        fmParallel,
        deps,
        2,
        d,
        core
    );
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
#ifdef _WIN32
    // Pin vcomp140.dll in memory so MSVC OpenMP worker threads remain valid upon FreeLibrary(vsflip.dll)
    HMODULE hVcomp = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, L"vcomp140.dll", &hVcomp);
#endif

    vspapi->configPlugin(
        "com.pingwer.vsflip",
        "vsflip",
        "VapourSynth FLIP Perceptual Metric",
        VS_MAKE_VERSION(2, 0),
        VAPOURSYNTH_API_VERSION,
        0,
        plugin
    );

    vspapi->registerFunction(
        "Evaluate",
        "reference:vnode;"
        "test:vnode;"
        "dynamic_range:int:opt;"
        "ppd:float:opt;"
        "vc:float[]:opt;"
        "tonemapper:data:opt;"
        "start_exposure:float:opt;"
        "stop_exposure:float:opt;"
        "num_exposures:int:opt;"
        "magma:int:opt;"
        "device:int:opt;"
        "cuda_device:int:opt;"
        "input_srgb:int:opt;",
        "clip:vnode;",
        flipCreate,
        nullptr,
        plugin
    );
}

