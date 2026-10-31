#include "flip_backend.h"

#ifdef VSFLIP_CUDA_SUPPORT
#include <cuda_runtime.h>

#define FLIP_ENABLE_CUDA 1
#define FLIP FLIP_CUDA
#include "include/FLIP.h"
#undef FLIP

#include <vector>
#include <algorithm>
#include <limits>
#include <cstring>
#include <stdexcept>

bool flip_cuda_is_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

int flip_cuda_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess) ? count : 0;
}

bool flip_evaluate_cuda(
    const float* ref_interleaved,
    const float* test_interleaved,
    int width,
    int height,
    const FlipEvaluationParams& params,
    int device_id,
    float* output_data,
    double& mean_error
) {
    if (!ref_interleaved || !test_interleaved || !output_data || width <= 0 || height <= 0) {
        return false;
    }

    int dev_count = flip_cuda_device_count();
    if (dev_count <= 0) {
        return false;
    }
    if (device_id < 0 || device_id >= dev_count) {
        device_id = 0;
    }
    cudaError_t setDevErr = cudaSetDevice(device_id);
    if (setDevErr != cudaSuccess) {
        return false;
    }

    try {
        size_t num_pixels = static_cast<size_t>(width) * height;

        FLIP_CUDA::Parameters flip_params;
        if (params.use_vc) {
            flip_params.PPD = FLIP_CUDA::calculatePPD(params.vc[0], params.vc[1], params.vc[2]);
        } else {
            flip_params.PPD = (params.ppd > 0.0f) ? params.ppd : 67.0f;
        }

        flip_params.tonemapper = params.tonemapper.empty() ? "aces" : params.tonemapper;
        flip_params.startExposure = params.start_exposure;
        flip_params.stopExposure = params.stop_exposure;
        flip_params.numExposures = params.num_exposures;

        bool use_hdr = (params.dynamic_range == 1);
        float mean_err_float = -1.0f;
        float* error_map_raw = nullptr;

        FLIP_CUDA::evaluate(
            ref_interleaved,
            test_interleaved,
            width,
            height,
            use_hdr,
            flip_params,
            params.apply_magma,
            true, // computeMeanFLIPError
            mean_err_float,
            &error_map_raw
        );

        mean_error = static_cast<double>(mean_err_float);

        if (error_map_raw != nullptr) {
            size_t out_floats = params.apply_magma ? (num_pixels * 3) : num_pixels;
            std::memcpy(output_data, error_map_raw, out_floats * sizeof(float));
            delete[] error_map_raw;
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "vsflip CUDA error: " << e.what() << "\n";
        return false;
    } catch (...) {
        std::cerr << "vsflip CUDA unknown error\n";
        return false;
    }

    return false;
}

#else

bool flip_cuda_is_available() {
    return false;
}

int flip_cuda_device_count() {
    return 0;
}

bool flip_evaluate_cuda(
    const float*,
    const float*,
    int,
    int,
    const FlipEvaluationParams&,
    int,
    float*,
    double&
) {
    return false;
}

#endif
