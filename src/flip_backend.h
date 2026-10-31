#pragma once
#include <cstddef>
#include <string>

enum class FlipDevice {
    CPU = 0,
    CUDA = 1,
    AUTO = 2
};

struct FlipEvaluationParams {
    int dynamic_range;          // 0 = LDR, 1 = HDR
    float ppd;                  // Pixels per degree (> 0)
    float vc[3];                // Viewing conditions [distance_m, width_px, width_m]
    bool use_vc;                // Whether to use vc instead of ppd
    std::string tonemapper;     // "aces", "hable", "reinhard"
    float start_exposure;       // HDR start exposure (infinity for auto)
    float stop_exposure;        // HDR stop exposure (infinity for auto)
    int num_exposures;          // HDR number of exposures (-1 for auto)
    bool apply_magma;           // false = grayscale (1 plane), true = magma heatmap (3 planes)
    bool input_srgb;            // true if inputs are sRGB (converted to linear RGB internally)
};

// Evaluates FLIP using CPU (OpenMP).
// ref_interleaved and test_interleaved are H x W x 3 float buffers.
// output_data must be preallocated: H x W for grayscale, or H x W x 3 for magma.
bool flip_evaluate_cpu(
    const float* ref_interleaved,
    const float* test_interleaved,
    int width,
    int height,
    const FlipEvaluationParams& params,
    float* output_data,
    double& mean_error
);

// Evaluates FLIP using CUDA on specified device_id.
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

// Returns true if CUDA is compiled and at least one CUDA-capable device is detected.
bool flip_cuda_is_available();

// Returns number of CUDA devices.
int flip_cuda_device_count();

