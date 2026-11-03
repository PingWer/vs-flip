#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

enum class FlipDevice {
    CPU = 0,
    CUDA = 1,
    AUTO = 2
};

struct FlipEvaluationParams {
    int dynamic_range;          // 0 = LDR, 1 = HDR
    float ppd;                  // pixels per degree
    float vc[3];                // dist_m, w_px, w_m
    bool use_vc;
    std::string tonemapper;
    float start_exposure;
    float stop_exposure;
    int num_exposures;
    bool apply_magma;           // false = 1 gray plane, true = 3 magma planes
    bool fast;                  // FP16 fast mode (default true in 2.1)
    bool gpu_output;            // true = write direct to GPU plane, false = write to host RAM
};

struct VulkanPlaneDesc {
    uint64_t memoryId = 0;
    uint64_t memorySize = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
    int handleType = 0;
    intptr_t handle = 0;
    int stride_bytes = 0;

    // device-side timeline semaphore sync
    uint64_t vkSemaphore = 0;
    uint64_t readyValue = 0;
    int semHandleType = 0;
    intptr_t semHandle = 0;
};

// cpu openmp path
bool flip_evaluate_cpu(
    const float* ref_interleaved,
    const float* test_interleaved,
    int width,
    int height,
    const FlipEvaluationParams& params,
    float* output_data,
    double& mean_error
);

// cuda path from host ram (interleaved)
bool flip_evaluate_cuda(
    const float* ref_interleaved,
    const float* test_interleaved,
    int width,
    int height,
    const FlipEvaluationParams& params,
    int device_id,
    float* output_data,
    double& mean_error
);

// cuda path from host ram
bool flip_evaluate_cuda_planar(
    const uint8_t* const ref_ptrs[3],
    const ptrdiff_t ref_strides[3],
    const uint8_t* const test_ptrs[3],
    const ptrdiff_t test_strides[3],
    int width,
    int height,
    const FlipEvaluationParams& params,
    int device_id,
    uint8_t* const out_ptrs[3],
    const ptrdiff_t out_strides[3],
    double& mean_error
);

// zero-copy: reads direct from vulkan vram, writes out to host ram
bool flip_evaluate_cuda_vulkan(
    const VulkanPlaneDesc ref_planes[3],
    const VulkanPlaneDesc test_planes[3],
    int width,
    int height,
    const FlipEvaluationParams& params,
    int device_id,
    uint8_t* const out_ptrs[3],
    const ptrdiff_t out_strides[3],
    double& mean_error
);

// unified flexible cuda input/output
bool flip_evaluate_cuda_flexible(
    const VulkanPlaneDesc* ref_vulkan,
    const uint8_t* const ref_host_ptrs[3],
    const ptrdiff_t ref_host_strides[3],
    const VulkanPlaneDesc* test_vulkan,
    const uint8_t* const test_host_ptrs[3],
    const ptrdiff_t test_host_strides[3],
    int width,
    int height,
    const FlipEvaluationParams& params,
    int device_id,
    uint8_t* const out_ptrs[3],
    const ptrdiff_t out_strides[3],
    double& mean_error
);

void* flip_cuda_import_vulkan_plane(const VulkanPlaneDesc& desc);

void flip_cuda_cleanup_memory_cache();
bool flip_cuda_is_available();
int flip_cuda_device_count();
