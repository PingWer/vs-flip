#include "flip_backend.h"

#define FLIP FLIP_CPU
#include "include/FLIP.h"
#undef FLIP

#include <vector>
#include <algorithm>
#include <limits>
#include <cstring>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

bool flip_evaluate_cpu(
    const float* ref_interleaved,
    const float* test_interleaved,
    int width,
    int height,
    const FlipEvaluationParams& params,
    float* output_data,
    double& mean_error
) {
    if (!ref_interleaved || !test_interleaved || !output_data || width <= 0 || height <= 0) {
        return false;
    }

#ifdef _OPENMP
    // Limit OpenMP threads inside each frame to 1.
    // VapourSynth already parallelizes across frames with fmParallel.
    // Nested OpenMP threads in MSVC's vcomp140.dll cause 0xC0000005 access violations
    // and severe CPU cache thrashing.
    omp_set_num_threads(1);
#endif

    try {
        size_t num_pixels = static_cast<size_t>(width) * height;

        FLIP_CPU::Parameters flip_params;
        if (params.use_vc) {
            flip_params.PPD = FLIP_CPU::calculatePPD(params.vc[0], params.vc[1], params.vc[2]);
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

        FLIP_CPU::evaluate(
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
        std::cerr << "vsflip CPU error: " << e.what() << "\n";
        return false;
    } catch (...) {
        std::cerr << "vsflip CPU unknown error\n";
        return false;
    }

    return false;
}
