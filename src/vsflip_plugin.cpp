#ifndef VS_USE_LATEST_API
#define VS_USE_LATEST_API
#endif
#include "include/VapourSynth4.h"
#include "include/VSHelper4.h"
#ifdef VSFLIP_HAS_VULKAN
#include <vulkan/vulkan_core.h>
#ifndef VK_API_VERSION_1_4
#define VK_API_VERSION_1_4 VK_MAKE_API_VERSION(0, 1, 4, 0)
#endif
#ifndef VK_VERSION_1_4
typedef void (*PFN_vkMapMemory2)(void);
typedef void (*PFN_vkUnmapMemory2)(void);
typedef void (*PFN_vkGetImageSubresourceLayout2)(void);
typedef void (*PFN_vkCmdBindDescriptorSets2)(void);
typedef void (*PFN_vkCmdPushConstants2)(void);
typedef void (*PFN_vkCmdPushDescriptorSet)(void);
typedef void (*PFN_vkCmdPushDescriptorSetWithTemplate)(void);
#endif
#include "include/VSVulkan4.h"
#endif
#include "flip_backend.h"

#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <limits>
#include <algorithm>
#include <iostream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

static inline bool is_api_43_or_higher(const VSAPI* vsapi) noexcept {
    if (!vsapi) return false;
    int ver = vsapi->getAPIVersion();
    int major = (ver >> 16) & 0xFFFF;
    int minor = ver & 0xFFFF;
    return (major > 4) || (major == 4 && minor >= 3);
}

static inline bool is_gpu_node(VSNode* node, const VSAPI* vsapi) noexcept {
    if (!node || !vsapi) return false;
#if VAPOURSYNTH_API_MINOR >= 3
    if (is_api_43_or_higher(vsapi) && vsapi->getNodeResidency) {
        return vsapi->getNodeResidency(node) == nrGPU;
    }
#endif
    return false;
}

struct FlipFilterData {
    VSNode* ref_node = nullptr;
    VSNode* test_node = nullptr;
    VSVideoInfo vi = {};
    FlipEvaluationParams params = {};
    FlipDevice device = FlipDevice::AUTO;
    int cuda_device_id = 0;
    bool ref_is_gpu = false;
    bool test_is_gpu = false;
    bool can_export_semaphore = false;
    bool can_gpu_output = false;
};

static void close_vulkan_plane_handles(VulkanPlaneDesc planes[3]) {
    for (int i = 0; i < 3; ++i) {
#ifdef _WIN32
        if (planes[i].handle) { CloseHandle(reinterpret_cast<HANDLE>(planes[i].handle)); planes[i].handle = 0; }
#else
        if (planes[i].handle > 0) { close(static_cast<int>(planes[i].handle)); planes[i].handle = 0; }
#endif
    }
#ifdef _WIN32
    if (planes[0].semHandle) { CloseHandle(reinterpret_cast<HANDLE>(planes[0].semHandle)); planes[0].semHandle = 0; }
#else
    if (planes[0].semHandle > 0) { close(static_cast<int>(planes[0].semHandle)); planes[0].semHandle = 0; }
#endif
}

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
        int width = d->vi.width;
        int height = d->vi.height;

        const VSFrame* ref_frame = nullptr;
        VulkanPlaneDesc ref_vk_planes[3] = {};
        bool ref_vk_valid = false;

        const VSFrame* test_frame = nullptr;
        VulkanPlaneDesc test_vk_planes[3] = {};
        bool test_vk_valid = false;

#if defined(VSFLIP_CUDA_SUPPORT) && defined(VSFLIP_HAS_VULKAN)
        const VSVULKANAPI* vkapi = vsapi->getVulkanAPI ? vsapi->getVulkanAPI() : nullptr;
        if (vkapi) {
            char err_msg[512] = {0};

            if (d->ref_is_gpu) {
                ref_frame = vkapi->getExportableFrameFilter(n, d->ref_node, frameCtx, err_msg, sizeof(err_msg));
                if (ref_frame) {
                    VSVulkanPlaneInfo r_info = {};
                    vkapi->getGPUPlane(ref_frame, 0, &r_info);

                    bool sem_synced = false;
                    if (d->can_export_semaphore && r_info.readySemaphore && r_info.readyValue > 0) {
                        VSVulkanExportedSemaphore exp_r = {};
                        if (vkapi->exportGPUSemaphore(core, r_info.readySemaphore, &exp_r, err_msg, sizeof(err_msg)) == 0) {
                            ref_vk_planes[0].vkSemaphore = reinterpret_cast<uint64_t>(r_info.readySemaphore);
                            ref_vk_planes[0].readyValue = r_info.readyValue;
                            ref_vk_planes[0].semHandleType = exp_r.handleType;
                            ref_vk_planes[0].semHandle = exp_r.handle;
                            sem_synced = true;
                        }
                    }
                    if (!sem_synced && r_info.readySemaphore) {
                        vkapi->waitGPUFrame(ref_frame, err_msg, sizeof(err_msg));
                    }

                    bool export_ok = true;
                    for (int p = 0; p < 3; ++p) {
                        VSVulkanExportedMemory exp_ref = {};
                        if (vkapi->exportGPUPlane(ref_frame, p, &exp_ref, err_msg, sizeof(err_msg)) != 0) {
                            export_ok = false;
                            break;
                        }
                        ref_vk_planes[p].memoryId = exp_ref.memoryId;
                        ref_vk_planes[p].memorySize = exp_ref.memorySize;
                        ref_vk_planes[p].offset = exp_ref.offset;
                        ref_vk_planes[p].size = exp_ref.size;
                        ref_vk_planes[p].handleType = exp_ref.handleType;
                        ref_vk_planes[p].handle = exp_ref.handle;
                        ref_vk_planes[p].stride_bytes = vsapi->getStride(ref_frame, p);
                    }
                    if (export_ok) {
                        ref_vk_valid = true;
                    } else {
                        close_vulkan_plane_handles(ref_vk_planes);
                    }
                }
            }

            // automatic test frame acquisition (GPU zero-copy or host)
            if (d->test_is_gpu) {
                test_frame = vkapi->getExportableFrameFilter(n, d->test_node, frameCtx, err_msg, sizeof(err_msg));
                if (test_frame) {
                    VSVulkanPlaneInfo t_info = {};
                    vkapi->getGPUPlane(test_frame, 0, &t_info);

                    bool sem_synced = false;
                    if (d->can_export_semaphore && t_info.readySemaphore && t_info.readyValue > 0) {
                        VSVulkanExportedSemaphore exp_t = {};
                        if (vkapi->exportGPUSemaphore(core, t_info.readySemaphore, &exp_t, err_msg, sizeof(err_msg)) == 0) {
                            test_vk_planes[0].vkSemaphore = reinterpret_cast<uint64_t>(t_info.readySemaphore);
                            test_vk_planes[0].readyValue = t_info.readyValue;
                            test_vk_planes[0].semHandleType = exp_t.handleType;
                            test_vk_planes[0].semHandle = exp_t.handle;
                            sem_synced = true;
                        }
                    }
                    if (!sem_synced && t_info.readySemaphore) {
                        vkapi->waitGPUFrame(test_frame, err_msg, sizeof(err_msg));
                    }

                    bool export_ok = true;
                    for (int p = 0; p < 3; ++p) {
                        VSVulkanExportedMemory exp_test = {};
                        if (vkapi->exportGPUPlane(test_frame, p, &exp_test, err_msg, sizeof(err_msg)) != 0) {
                            export_ok = false;
                            break;
                        }
                        test_vk_planes[p].memoryId = exp_test.memoryId;
                        test_vk_planes[p].memorySize = exp_test.memorySize;
                        test_vk_planes[p].offset = exp_test.offset;
                        test_vk_planes[p].size = exp_test.size;
                        test_vk_planes[p].handleType = exp_test.handleType;
                        test_vk_planes[p].handle = exp_test.handle;
                        test_vk_planes[p].stride_bytes = vsapi->getStride(test_frame, p);
                    }
                    if (export_ok) {
                        test_vk_valid = true;
                    } else {
                        close_vulkan_plane_handles(test_vk_planes);
                    }
                }
            }
        }
#endif

        if (!ref_frame) {
            ref_frame = vsapi->getFrameFilter(n, d->ref_node, frameCtx);
            if (!ref_frame) {
                if (test_frame) vsapi->freeFrame(test_frame);
                if (test_vk_valid) close_vulkan_plane_handles(test_vk_planes);
                vsapi->setFilterError("vsflip: failed to retrieve reference frame", frameCtx);
                return nullptr;
            }
        }
        if (!test_frame) {
            test_frame = vsapi->getFrameFilter(n, d->test_node, frameCtx);
            if (!test_frame) {
                if (ref_frame) vsapi->freeFrame(ref_frame);
                if (ref_vk_valid) close_vulkan_plane_handles(ref_vk_planes);
                vsapi->setFilterError("vsflip: failed to retrieve test frame", frameCtx);
                return nullptr;
            }
        }

        VSFrame* out_frame = nullptr;
        uint8_t* out_ptrs[3] = {};
        ptrdiff_t out_strides[3] = {};
        int num_out_planes = d->params.apply_magma ? 3 : 1;

#if defined(VSFLIP_CUDA_SUPPORT) && defined(VSFLIP_HAS_VULKAN)
        if (d->params.gpu_output && vkapi) {
            out_frame = vkapi->newGPUVideoFrame(&d->vi.format, width, height, ref_frame, core);
            if (!out_frame) {
                if (ref_vk_valid) close_vulkan_plane_handles(ref_vk_planes);
                if (test_vk_valid) close_vulkan_plane_handles(test_vk_planes);
                vsapi->freeFrame(ref_frame);
                vsapi->freeFrame(test_frame);
                vsapi->setFilterError("vsflip: failed to allocate GPU output frame", frameCtx);
                return nullptr;
            }

            char err_msg[512] = {0};
            bool out_export_ok = true;
            VulkanPlaneDesc out_vk_planes[3] = {};

            for (int p = 0; p < num_out_planes; ++p) {
                VSVulkanExportedMemory exp_out = {};
                if (vkapi->exportGPUPlane(out_frame, p, &exp_out, err_msg, sizeof(err_msg)) != 0) {
                    out_export_ok = false;
                    break;
                }
                out_vk_planes[p].memoryId = exp_out.memoryId;
                out_vk_planes[p].memorySize = exp_out.memorySize;
                out_vk_planes[p].offset = exp_out.offset;
                out_vk_planes[p].size = exp_out.size;
                out_vk_planes[p].handleType = exp_out.handleType;
                out_vk_planes[p].handle = exp_out.handle;
                out_vk_planes[p].stride_bytes = vsapi->getStride(out_frame, p);

                void* dev_ptr = flip_cuda_import_vulkan_plane(out_vk_planes[p]);
                if (!dev_ptr) {
                    out_export_ok = false;
                    break;
                }
                out_ptrs[p] = static_cast<uint8_t*>(dev_ptr);
                out_strides[p] = out_vk_planes[p].stride_bytes;
            }

            if (!out_export_ok) {
                close_vulkan_plane_handles(out_vk_planes);
                if (ref_vk_valid) close_vulkan_plane_handles(ref_vk_planes);
                if (test_vk_valid) close_vulkan_plane_handles(test_vk_planes);
                vsapi->freeFrame(out_frame);
                vsapi->freeFrame(ref_frame);
                vsapi->freeFrame(test_frame);
                vsapi->setFilterError("vsflip: failed to export GPU output plane", frameCtx);
                return nullptr;
            }
        }
#endif

        if (!out_frame) {
            out_frame = vsapi->newVideoFrame(&d->vi.format, width, height, ref_frame, core);
            if (!out_frame) {
                if (ref_vk_valid) close_vulkan_plane_handles(ref_vk_planes);
                if (test_vk_valid) close_vulkan_plane_handles(test_vk_planes);
                vsapi->freeFrame(ref_frame);
                vsapi->freeFrame(test_frame);
                vsapi->setFilterError("vsflip: failed to allocate CPU output frame", frameCtx);
                return nullptr;
            }
            for (int p = 0; p < num_out_planes; ++p) {
                out_ptrs[p] = vsapi->getWritePtr(out_frame, p);
                out_strides[p] = vsapi->getStride(out_frame, p);
            }
        }

        double mean_error = -1.0;
        bool ok = false;

        bool use_cuda = (d->device == FlipDevice::CUDA) ||
                        (d->device == FlipDevice::AUTO && flip_cuda_is_available());

        if (use_cuda) {
            const uint8_t* ref_host_ptrs[3] = { nullptr, nullptr, nullptr };
            ptrdiff_t ref_host_strides[3] = { 0, 0, 0 };
            if (!ref_vk_valid) {
                for (int p = 0; p < 3; ++p) {
                    ref_host_ptrs[p] = vsapi->getReadPtr(ref_frame, p);
                    ref_host_strides[p] = vsapi->getStride(ref_frame, p);
                }
            }

            const uint8_t* test_host_ptrs[3] = { nullptr, nullptr, nullptr };
            ptrdiff_t test_host_strides[3] = { 0, 0, 0 };
            if (!test_vk_valid) {
                for (int p = 0; p < 3; ++p) {
                    test_host_ptrs[p] = vsapi->getReadPtr(test_frame, p);
                    test_host_strides[p] = vsapi->getStride(test_frame, p);
                }
            }

            ok = flip_evaluate_cuda_flexible(
                ref_vk_valid ? ref_vk_planes : nullptr,
                ref_vk_valid ? nullptr : ref_host_ptrs,
                ref_host_strides,
                test_vk_valid ? test_vk_planes : nullptr,
                test_vk_valid ? nullptr : test_host_ptrs,
                test_host_strides,
                width,
                height,
                d->params,
                d->cuda_device_id,
                out_ptrs,
                out_strides,
                mean_error
            );
        }

        if (!ok && !d->params.gpu_output) {
            // CPU fallback (only allowed when gpu_output is false)
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

            size_t out_floats = d->params.apply_magma ? (num_pixels * 3) : num_pixels;
            std::vector<float> output_data(out_floats, 0.0f);

            ok = flip_evaluate_cpu(
                ref_interleaved.data(),
                test_interleaved.data(),
                width,
                height,
                d->params,
                output_data.data(),
                mean_error
            );

            if (ok) {
                if (d->params.apply_magma) {
                    for (int y = 0; y < height; ++y) {
                        auto* dr = reinterpret_cast<float*>(out_ptrs[0] + y * out_strides[0]);
                        auto* dg = reinterpret_cast<float*>(out_ptrs[1] + y * out_strides[1]);
                        auto* db = reinterpret_cast<float*>(out_ptrs[2] + y * out_strides[2]);
                        const float* src_row = output_data.data() + static_cast<size_t>(y) * width * 3;

                        for (int x = 0; x < width; ++x) {
                            dr[x] = src_row[x * 3 + 0];
                            dg[x] = src_row[x * 3 + 1];
                            db[x] = src_row[x * 3 + 2];
                        }
                    }
                } else {
                    for (int y = 0; y < height; ++y) {
                        auto* dst_row = reinterpret_cast<float*>(out_ptrs[0] + y * out_strides[0]);
                        const float* src_row = output_data.data() + static_cast<size_t>(y) * width;
                        std::memcpy(dst_row, src_row, width * sizeof(float));
                    }
                }
            }
        }

        if (!ok) {
            vsapi->setFilterError("vsflip: metric evaluation failed", frameCtx);
            vsapi->freeFrame(out_frame);
            vsapi->freeFrame(ref_frame);
            vsapi->freeFrame(test_frame);
            return nullptr;
        }

#if defined(VSFLIP_CUDA_SUPPORT) && defined(VSFLIP_HAS_VULKAN)
        if (d->params.gpu_output && vkapi) {
            for (int p = 0; p < num_out_planes; ++p) {
                vkapi->setGPUPlaneProducer(out_frame, p, nullptr, 0);
            }
        }
#endif

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
        flip_cuda_cleanup_memory_cache();
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

    d->params.dynamic_range = static_cast<int>(vsapi->mapGetInt(in, "dynamic_range", 0, &err));
    if (err) d->params.dynamic_range = 0;

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

    int fast_opt = static_cast<int>(vsapi->mapGetInt(in, "fast", 0, &err));
    if (err) {
        d->params.fast = true;
    } else {
        d->params.fast = (fast_opt != 0);
    }

    int superfast_opt = static_cast<int>(vsapi->mapGetInt(in, "superfast", 0, &err));
    if (!err && superfast_opt != 0) {
        d->params.fast = true;
    }

    int gpu_out_opt = static_cast<int>(vsapi->mapGetInt(in, "gpu_output", 0, &err));
    d->params.gpu_output = (!err && gpu_out_opt != 0);

    if (d->params.apply_magma) {
        vsapi->queryVideoFormat(&d->vi.format, cfRGB, stFloat, 32, 0, 0, core);
    } else {
        vsapi->queryVideoFormat(&d->vi.format, cfGray, stFloat, 32, 0, 0, core);
    }

    VSFilterDependency deps[] = {
        { d->ref_node, rpStrictSpatial },
        { d->test_node, rpStrictSpatial }
    };

    bool use_cuda = (d->device == FlipDevice::CUDA) ||
                    (d->device == FlipDevice::AUTO && flip_cuda_is_available());

#if defined(VSFLIP_CUDA_SUPPORT) && defined(VSFLIP_HAS_VULKAN)
    const VSVULKANAPI* vkapi = is_api_43_or_higher(vsapi) && vsapi->getVulkanAPI ? vsapi->getVulkanAPI() : nullptr;

    bool vulkan_export_supported = false;
    bool vulkan_sem_export_supported = false;
    if (vkapi) {
        VSVulkanCoreInfo core_info = {};
        char err_buf[256] = {0};
        if (vkapi->getVulkanCoreInfo(core, &core_info, err_buf, sizeof(err_buf)) == 0) {
            vulkan_export_supported = (core_info.exportHandleType != 0);
            vulkan_sem_export_supported = (core_info.semaphoreExportHandleType != 0);
        }
    }

    d->ref_is_gpu = use_cuda && is_gpu_node(d->ref_node, vsapi) && vkapi && vulkan_export_supported;
    d->test_is_gpu = use_cuda && is_gpu_node(d->test_node, vsapi) && vkapi && vulkan_export_supported;
    d->can_export_semaphore = (d->ref_is_gpu || d->test_is_gpu) && vulkan_sem_export_supported;
    d->can_gpu_output = use_cuda && d->params.gpu_output && vkapi && vulkan_export_supported;
#else
    d->ref_is_gpu = false;
    d->test_is_gpu = false;
    d->can_export_semaphore = false;
    d->can_gpu_output = false;
#endif

    if (d->params.gpu_output) {
#if defined(VSFLIP_CUDA_SUPPORT) && defined(VSFLIP_HAS_VULKAN) && (VAPOURSYNTH_API_MINOR >= 3)
        if (!d->can_gpu_output || !vsapi->createVideoFilterEx) {
            vsapi->freeNode(ref_node);
            vsapi->freeNode(test_node);
            delete d;
            vsapi->mapSetError(out, "vsflip: 'gpu_output=True' requires VapourSynth API 4.3+ with Vulkan and CUDA support.");
            return;
        }
        vsapi->createVideoFilterEx(
            out,
            "Evaluate",
            &d->vi,
            flipGetFrame,
            flipFree,
            fmParallel,
            ffGPUOutput,
            deps,
            2,
            d,
            core
        );
        return;
#else
        vsapi->freeNode(ref_node);
        vsapi->freeNode(test_node);
        delete d;
        vsapi->mapSetError(out, "vsflip: 'gpu_output=True' is not supported in this build.");
        return;
#endif
    }

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
    // pin vcomp140 in memory so FreeLibrary doesnt crash msvc openmp threads
    HMODULE hVcomp = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, L"vcomp140.dll", &hVcomp);
#endif

    vspapi->configPlugin(
        "com.pingwer.vsflip",
        "vsflip",
        "VapourSynth FLIP Perceptual Metric",
        VS_MAKE_VERSION(2, 1),
        VAPOURSYNTH_API_VERSION,
        0,
        plugin
    );

    bool api_43_or_higher = false;
    if (vspapi && vspapi->getAPIVersion) {
        int ver = vspapi->getAPIVersion();
        int major = (ver >> 16) & 0xFFFF;
        int minor = ver & 0xFFFF;
        api_43_or_higher = (major > 4) || (major == 4 && minor >= 3);
    }

    const char* args = api_43_or_higher ?
        "reference:vnode:all;"
        "test:vnode:all;"
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
        "fast:int:opt;"
        "superfast:int:opt;"
        "gpu_output:int:opt;"
        :
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
        "fast:int:opt;"
        "superfast:int:opt;"
        "gpu_output:int:opt;";

    const char* ret_sig = api_43_or_higher ? "clip:vnode:all;" : "clip:vnode;";

    vspapi->registerFunction(
        "Evaluate",
        args,
        ret_sig,
        flipCreate,
        nullptr,
        plugin
    );
}
