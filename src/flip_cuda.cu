#include "flip_backend.h"

#ifdef VSFLIP_CUDA_SUPPORT
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define FLIP_ENABLE_CUDA 1
#define FLIP FLIP_CUDA
#include "include/FLIP.h"
#undef FLIP

#include <vector>
#include <algorithm>
#include <limits>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <cstdlib>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

// cached vulkan device memory allocations
struct CachedVulkanAllocation {
    cudaExternalMemory_t extMem = nullptr;
    void* baseDevPtr = nullptr;
    uint64_t memorySize = 0;
    uint64_t lastAccess = 0;
};

static std::unordered_map<uint64_t, CachedVulkanAllocation> s_vk_alloc_cache;
static std::mutex s_vk_cache_mutex;
static uint64_t s_vk_frame_counter = 0;
static constexpr size_t MAX_VK_CACHE_ENTRIES = 64;

// cached vulkan timeline semaphores
struct CachedVulkanSemaphore {
    cudaExternalSemaphore_t extSem = nullptr;
    uint64_t lastAccess = 0;
};
static std::unordered_map<uint64_t, CachedVulkanSemaphore> s_vk_sem_cache;
static std::mutex s_vk_sem_cache_mutex;

static cudaExternalSemaphore_t get_or_import_vulkan_semaphore(uint64_t vkSem, int handleType, intptr_t handle) {
    if (!vkSem || !handle) return nullptr;

    std::lock_guard<std::mutex> lock(s_vk_sem_cache_mutex);
    auto it = s_vk_sem_cache.find(vkSem);
    if (it != s_vk_sem_cache.end()) {
#ifdef _WIN32
        if (handle) CloseHandle(reinterpret_cast<HANDLE>(handle));
#else
        if (handle > 0) close(static_cast<int>(handle));
#endif
        it->second.lastAccess = s_vk_frame_counter;
        return it->second.extSem;
    }

    if (s_vk_sem_cache.size() >= 32) {
        auto oldest = s_vk_sem_cache.begin();
        for (auto cur = s_vk_sem_cache.begin(); cur != s_vk_sem_cache.end(); ++cur) {
            if (cur->second.lastAccess < oldest->second.lastAccess) oldest = cur;
        }
        if (oldest != s_vk_sem_cache.end()) {
            if (oldest->second.extSem) cudaDestroyExternalSemaphore(oldest->second.extSem);
            s_vk_sem_cache.erase(oldest);
        }
    }

    cudaExternalSemaphoreHandleDesc desc = {};
#ifdef _WIN32
    desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    desc.handle.win32.handle = reinterpret_cast<void*>(handle);
#else
    desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    desc.handle.fd = static_cast<int>(handle);
#endif

    cudaExternalSemaphore_t extSem = nullptr;
    cudaError_t err = cudaImportExternalSemaphore(&extSem, &desc);

#ifdef _WIN32
    if (handle) CloseHandle(reinterpret_cast<HANDLE>(handle));
#else
    if (err != cudaSuccess && handle > 0) close(static_cast<int>(handle));
#endif

    if (err != cudaSuccess || !extSem) return nullptr;

    CachedVulkanSemaphore entry;
    entry.extSem = extSem;
    entry.lastAccess = s_vk_frame_counter;
    s_vk_sem_cache[vkSem] = entry;
    return extSem;
}

static void* get_or_import_vulkan_plane(const VulkanPlaneDesc& desc) {
    if (desc.memorySize == 0 || desc.size == 0) return nullptr;
    if (desc.offset + desc.size > desc.memorySize) return nullptr;

    std::lock_guard<std::mutex> lock(s_vk_cache_mutex);
    s_vk_frame_counter++;

    auto it = s_vk_alloc_cache.find(desc.memoryId);
    if (it != s_vk_alloc_cache.end()) {
#ifdef _WIN32
        if (desc.handle) CloseHandle((HANDLE)desc.handle);
#else
        if (desc.handle > 0) close((int)desc.handle);
#endif
        it->second.lastAccess = s_vk_frame_counter;
        return static_cast<uint8_t*>(it->second.baseDevPtr) + desc.offset;
    }

    if (s_vk_alloc_cache.size() >= MAX_VK_CACHE_ENTRIES) {
        auto oldest_it = s_vk_alloc_cache.begin();
        for (auto cur = s_vk_alloc_cache.begin(); cur != s_vk_alloc_cache.end(); ++cur) {
            if (cur->second.lastAccess < oldest_it->second.lastAccess) {
                oldest_it = cur;
            }
        }
        if (oldest_it != s_vk_alloc_cache.end()) {
            if (oldest_it->second.baseDevPtr) cudaFree(oldest_it->second.baseDevPtr);
            if (oldest_it->second.extMem) cudaDestroyExternalMemory(oldest_it->second.extMem);
            s_vk_alloc_cache.erase(oldest_it);
        }
    }

    cudaExternalMemoryHandleDesc handleDesc = {};
#ifdef _WIN32
    if (desc.handleType & 0x10) {
        handleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
    } else {
        handleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    }
    handleDesc.handle.win32.handle = (void*)desc.handle;
#else
    handleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    handleDesc.handle.fd = (int)desc.handle;
#endif
    handleDesc.size = desc.memorySize;

    cudaExternalMemory_t extMem = nullptr;
    cudaError_t err = cudaImportExternalMemory(&extMem, &handleDesc);

#ifdef _WIN32
    if (desc.handle) CloseHandle((HANDLE)desc.handle);
#else
    if (desc.handle > 0) close((int)desc.handle);
#endif

    if (err != cudaSuccess || !extMem) {
        return nullptr;
    }

    cudaExternalMemoryBufferDesc bufDesc = {};
    bufDesc.offset = 0;
    bufDesc.size = desc.memorySize;
    void* baseDevPtr = nullptr;
    cudaError_t mapErr = cudaExternalMemoryGetMappedBuffer(&baseDevPtr, extMem, &bufDesc);
    if (mapErr != cudaSuccess || !baseDevPtr) {
        cudaDestroyExternalMemory(extMem);
        return nullptr;
    }

    CachedVulkanAllocation entry;
    entry.extMem = extMem;
    entry.baseDevPtr = baseDevPtr;
    entry.memorySize = desc.memorySize;
    entry.lastAccess = s_vk_frame_counter;
    s_vk_alloc_cache[desc.memoryId] = entry;

    return static_cast<uint8_t*>(baseDevPtr) + desc.offset;
}

void* flip_cuda_import_vulkan_plane(const VulkanPlaneDesc& desc) {
    return get_or_import_vulkan_plane(desc);
}

struct FastCudaWorkspace {
    void* d_workspace = nullptr;
    size_t allocated_size = 0;
    cudaStream_t stream = nullptr;
    double* h_pinned_sum = nullptr;

    std::unique_ptr<FLIP_CUDA::image<FLIP_CUDA::color3>> spatialFilterYCx;
    std::unique_ptr<FLIP_CUDA::image<FLIP_CUDA::color3>> spatialFilterCz;
    std::unique_ptr<FLIP_CUDA::image<FLIP_CUDA::color3>> featureFilter;
    int spatial_filter_radius = -1;
    int feature_filter_radius = -1;
    float current_ppd = -1.0f;

    FastCudaWorkspace() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cudaMallocHost(&h_pinned_sum, sizeof(double));
        if (h_pinned_sum) *h_pinned_sum = 0.0;
    }

    ~FastCudaWorkspace() {
        if (d_workspace) cudaFree(d_workspace);
        if (h_pinned_sum) cudaFreeHost(h_pinned_sum);
        if (stream) cudaStreamDestroy(stream);
    }

    bool ensure_size(size_t required_size) {
        if (allocated_size >= required_size && d_workspace != nullptr) {
            return true;
        }
        if (d_workspace) {
            cudaFree(d_workspace);
            d_workspace = nullptr;
            allocated_size = 0;
        }
        if (cudaMalloc(&d_workspace, required_size) != cudaSuccess) {
            return false;
        }
        allocated_size = required_size;
        return true;
    }

    bool ensure_filters(float ppd) {
        if (current_ppd == ppd && spatial_filter_radius > 0 && spatialFilterYCx && spatialFilterCz && featureFilter) {
            return true;
        }
        current_ppd = ppd;
        spatial_filter_radius = FLIP_CUDA::calculateSpatialFilterRadius(ppd);
        int spatialWidth = 2 * spatial_filter_radius + 1;
        spatialFilterYCx = std::make_unique<FLIP_CUDA::image<FLIP_CUDA::color3>>(spatialWidth, 1);
        spatialFilterCz = std::make_unique<FLIP_CUDA::image<FLIP_CUDA::color3>>(spatialWidth, 1);
        FLIP_CUDA::image<FLIP_CUDA::color3>::setSpatialFilters(*spatialFilterYCx, *spatialFilterCz, ppd, spatial_filter_radius);
        spatialFilterYCx->synchronizeDevice();
        spatialFilterCz->synchronizeDevice();

        const float stdDev = 0.5f * FLIP_CUDA::FLIPConstants.gw * ppd;
        feature_filter_radius = int(std::ceil(3.0f * stdDev));
        int featureWidth = 2 * feature_filter_radius + 1;
        featureFilter = std::make_unique<FLIP_CUDA::image<FLIP_CUDA::color3>>(featureWidth, 1);
        FLIP_CUDA::image<FLIP_CUDA::color3>::setFeatureFilter(*featureFilter, ppd);
        featureFilter->synchronizeDevice();
        return true;
    }
};

class FastCudaWorkspacePool {
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::vector<std::unique_ptr<FastCudaWorkspace>> m_pool;
    int m_active_count = 0;

public:
    std::unique_ptr<FastCudaWorkspace> acquire(size_t num_pixels) {
        // Concurrency throttler to prevent GPU Out Of Memory on 4K/8K:
        // 8K (~33M px): 1 active frame saturates the entire GPU (SMs & memory bandwidth).
        // 4K (~8.3M px): max 2 active frames allow double-buffering without VRAM thrashing.
        // <= 1080p: max 8 active frames.
        int max_active = 8;
        if (num_pixels >= 30000000) {
            max_active = 1;
        } else if (num_pixels >= 7000000) {
            max_active = 2;
        }

        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [&]() {
            return m_active_count < max_active;
        });

        m_active_count++;
        if (!m_pool.empty()) {
            auto ws = std::move(m_pool.back());
            m_pool.pop_back();
            return ws;
        }
        return std::make_unique<FastCudaWorkspace>();
    }

    void release(std::unique_ptr<FastCudaWorkspace> ws) {
        if (!ws) return;
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pool.push_back(std::move(ws));
        m_active_count--;
        m_cv.notify_one();
    }

    void cleanup() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pool.clear();
        m_active_count = 0;
        m_cv.notify_all();
    }
};

static FastCudaWorkspacePool s_workspace_pool;

// 256-entry Magma color map in device constant memory
static __constant__ float c_magma_lut[256 * 3];
static bool s_magma_lut_init_device[16] = { false };
static std::mutex s_magma_lut_mutex;

static void ensure_magma_lut_initialized(int device_id) {
    if (device_id < 0 || device_id >= 16) device_id = 0;
    if (s_magma_lut_init_device[device_id]) return;
    std::lock_guard<std::mutex> lock(s_magma_lut_mutex);
    if (s_magma_lut_init_device[device_id]) return;

    int prev_device = 0;
    cudaGetDevice(&prev_device);
    cudaSetDevice(device_id);

    float h_lut[256 * 3];
    for (int i = 0; i < 256; ++i) {
        h_lut[i * 3 + 0] = FLIP_CUDA::MapMagma[i].r;
        h_lut[i * 3 + 1] = FLIP_CUDA::MapMagma[i].g;
        h_lut[i * 3 + 2] = FLIP_CUDA::MapMagma[i].b;
    }
    cudaMemcpyToSymbol(c_magma_lut, h_lut, sizeof(h_lut));
    s_magma_lut_init_device[device_id] = true;

    cudaSetDevice(prev_device);
}

void flip_cuda_cleanup_memory_cache() {
    {
        std::lock_guard<std::mutex> lock(s_vk_cache_mutex);
        for (auto& pair : s_vk_alloc_cache) {
            if (pair.second.baseDevPtr) cudaFree(pair.second.baseDevPtr);
            if (pair.second.extMem) cudaDestroyExternalMemory(pair.second.extMem);
        }
        s_vk_alloc_cache.clear();
    }
    {
        std::lock_guard<std::mutex> lock(s_vk_sem_cache_mutex);
        for (auto& pair : s_vk_sem_cache) {
            if (pair.second.extSem) cudaDestroyExternalSemaphore(pair.second.extSem);
        }
        s_vk_sem_cache.clear();
    }
    s_workspace_pool.cleanup();
    {
        std::lock_guard<std::mutex> lock(s_magma_lut_mutex);
        std::fill(std::begin(s_magma_lut_init_device), std::end(s_magma_lut_init_device), false);
    }
}

__global__ void pack_planar_to_color3_kernel(
    const float* __restrict__ r_plane,
    const float* __restrict__ g_plane,
    const float* __restrict__ b_plane,
    int stride_elements,
    FLIP_CUDA::color3* __restrict__ dst,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int src_idx = y * stride_elements + x;
        int dst_idx = y * width + x;
        dst[dst_idx] = FLIP_CUDA::color3(r_plane[src_idx], g_plane[src_idx], b_plane[src_idx]);
    }
}

// Map 1-channel error map [0, 1] directly to planar RGBS outputs using constant memory LUT
__global__ void magma_map_planar_kernel(
    const float* __restrict__ error_map,
    float* __restrict__ r_plane,
    int r_stride_elements,
    float* __restrict__ g_plane,
    int g_stride_elements,
    float* __restrict__ b_plane,
    int b_stride_elements,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float err = error_map[y * width + x];
        int idx = static_cast<int>(err * 255.0f + 0.5f);
        if (idx < 0) idx = 0;
        idx %= 256;

        r_plane[y * r_stride_elements + x] = c_magma_lut[idx * 3 + 0];
        g_plane[y * g_stride_elements + x] = c_magma_lut[idx * 3 + 1];
        b_plane[y * b_stride_elements + x] = c_magma_lut[idx * 3 + 2];
    }
}

// Map 1-channel error map [0, 1] to packed interleaved RGBS float array
__global__ void magma_map_interleaved_kernel(
    const float* __restrict__ error_map,
    float* __restrict__ dst_interleaved,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        float err = error_map[y * width + x];
        int idx = static_cast<int>(err * 255.0f + 0.5f);
        if (idx < 0) idx = 0;
        idx %= 256;

        int out_idx = (y * width + x) * 3;
        dst_interleaved[out_idx + 0] = c_magma_lut[idx * 3 + 0];
        dst_interleaved[out_idx + 1] = c_magma_lut[idx * 3 + 1];
        dst_interleaved[out_idx + 2] = c_magma_lut[idx * 3 + 2];
    }
}

// parallel sum on gpu with warp shuffle
template <unsigned int blockSize>
__global__ void fast_reduce_sum_kernel(
    const float* __restrict__ d_in,
    double* __restrict__ d_out,
    size_t total_elements
) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    size_t i = blockIdx.x * (blockSize * 2) + tid;
    size_t gridSize = blockSize * 2 * gridDim.x;

    double sum = 0.0;
    while (i < total_elements) {
        sum += static_cast<double>(d_in[i]);
        if (i + blockSize < total_elements) {
            sum += static_cast<double>(d_in[i + blockSize]);
        }
        i += gridSize;
    }
    sdata[tid] = sum;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  { sdata[tid] += sdata[tid + 64]; }  __syncthreads(); }
    if (blockSize >= 64)  { if (tid < 32)  { sdata[tid] += sdata[tid + 32]; }  __syncthreads(); }

    if (tid < 32) {
        double val = sdata[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) atomicAdd(d_out, val);
    }
}

static double compute_gpu_mean_stream(
    const float* d_error_map,
    size_t num_pixels,
    double* d_sum,
    cudaStream_t stream
) {
    const unsigned int blockSize = 256;
    unsigned int numBlocks = static_cast<unsigned int>((num_pixels + (blockSize * 2 - 1)) / (blockSize * 2));
    if (numBlocks > 1024) numBlocks = 1024;

    cudaMemsetAsync(d_sum, 0, sizeof(double), stream);
    fast_reduce_sum_kernel<blockSize><<<numBlocks, blockSize, blockSize * sizeof(double), stream>>>(d_error_map, d_sum, num_pixels);

    double h_sum = 0.0;
    cudaMemcpyAsync(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return h_sum / static_cast<double>(num_pixels);
}

static double compute_gpu_mean(const float* d_error_map, size_t num_pixels) {
    double* d_sum = nullptr;
    if (cudaMalloc(&d_sum, sizeof(double)) != cudaSuccess) return -1.0;
    double res = compute_gpu_mean_stream(d_error_map, num_pixels, d_sum, 0);
    cudaFree(d_sum);
    return res;
}

template <typename T>
struct RealTraits;

template <>
struct RealTraits<__half> {
    __host__ __device__ static inline __half from_float(float v) {
#if defined(__CUDA_ARCH__)
        return __float2half(v);
#else
        return __float2half(v);
#endif
    }
    __host__ __device__ static inline float to_float(__half v) {
#if defined(__CUDA_ARCH__)
        return __half2float(v);
#else
        return __half2float(v);
#endif
    }
};

template <>
struct RealTraits<float> {
    __host__ __device__ static inline float from_float(float v) { return v; }
    __host__ __device__ static inline float to_float(float v) { return v; }
};

template <typename T>
struct alignas(4) Vec2 {
    T x, y;
    __host__ __device__ Vec2() : x(RealTraits<T>::from_float(0.0f)), y(RealTraits<T>::from_float(0.0f)) {}
    __host__ __device__ Vec2(T _x, T _y) : x(_x), y(_y) {}
    __host__ __device__ Vec2(float _x, float _y) : x(RealTraits<T>::from_float(_x)), y(RealTraits<T>::from_float(_y)) {}
};

template <typename T>
struct Vec3 {
    T x, y, z;
    __host__ __device__ Vec3() : x(RealTraits<T>::from_float(0.0f)), y(RealTraits<T>::from_float(0.0f)), z(RealTraits<T>::from_float(0.0f)) {}
    __host__ __device__ Vec3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
    __host__ __device__ Vec3(float _x, float _y, float _z) : x(RealTraits<T>::from_float(_x)), y(RealTraits<T>::from_float(_y)), z(RealTraits<T>::from_float(_z)) {}
};

template <typename T>
__global__ void pack_and_convert_planar_to_ycxcz_kernel(
    const float* __restrict__ r_plane,
    const float* __restrict__ g_plane,
    const float* __restrict__ b_plane,
    int stride_elements,
    Vec3<T>* __restrict__ dst,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int src_idx = y * stride_elements + x;
        int dst_idx = y * width + x;
        float r = fminf(fmaxf(r_plane[src_idx], 0.0f), 1.0f);
        float g = fminf(fmaxf(g_plane[src_idx], 0.0f), 1.0f);
        float b = fminf(fmaxf(b_plane[src_idx], 0.0f), 1.0f);
        FLIP_CUDA::color3 rgb(r, g, b);
        FLIP_CUDA::color3 ycxcz = FLIP_CUDA::color3::XYZToYCxCz(FLIP_CUDA::color3::LinearRGBToXYZ(rgb));
        dst[dst_idx] = Vec3<T>(ycxcz.x, ycxcz.y, ycxcz.z);
    }
}

template <typename T>
__global__ void pack_and_convert_planar_pair_to_ycxcz_kernel(
    const float* __restrict__ ref_r,
    const float* __restrict__ ref_g,
    const float* __restrict__ ref_b,
    int ref_stride,
    Vec3<T>* __restrict__ dst_ref,
    const float* __restrict__ test_r,
    const float* __restrict__ test_g,
    const float* __restrict__ test_b,
    int test_stride,
    Vec3<T>* __restrict__ dst_test,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int dst_idx = y * width + x;

        int ref_idx = y * ref_stride + x;
        float rr = fminf(fmaxf(ref_r[ref_idx], 0.0f), 1.0f);
        float rg = fminf(fmaxf(ref_g[ref_idx], 0.0f), 1.0f);
        float rb = fminf(fmaxf(ref_b[ref_idx], 0.0f), 1.0f);
        FLIP_CUDA::color3 r_rgb(rr, rg, rb);
        FLIP_CUDA::color3 r_ycxcz = FLIP_CUDA::color3::XYZToYCxCz(FLIP_CUDA::color3::LinearRGBToXYZ(r_rgb));
        dst_ref[dst_idx] = Vec3<T>(r_ycxcz.x, r_ycxcz.y, r_ycxcz.z);

        int test_idx = y * test_stride + x;
        float tr = fminf(fmaxf(test_r[test_idx], 0.0f), 1.0f);
        float tg = fminf(fmaxf(test_g[test_idx], 0.0f), 1.0f);
        float tb = fminf(fmaxf(test_b[test_idx], 0.0f), 1.0f);
        FLIP_CUDA::color3 t_rgb(tr, tg, tb);
        FLIP_CUDA::color3 t_ycxcz = FLIP_CUDA::color3::XYZToYCxCz(FLIP_CUDA::color3::LinearRGBToXYZ(t_rgb));
        dst_test[dst_idx] = Vec3<T>(t_ycxcz.x, t_ycxcz.y, t_ycxcz.z);
    }
}

template <typename T>
__global__ void convert_interleaved_to_ycxcz_kernel(
    const float* __restrict__ src_interleaved,
    Vec3<T>* __restrict__ dst,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        int src_idx = idx * 3;
        float r = fminf(fmaxf(src_interleaved[src_idx + 0], 0.0f), 1.0f);
        float g = fminf(fmaxf(src_interleaved[src_idx + 1], 0.0f), 1.0f);
        float b = fminf(fmaxf(src_interleaved[src_idx + 2], 0.0f), 1.0f);
        FLIP_CUDA::color3 rgb(r, g, b);
        FLIP_CUDA::color3 ycxcz = FLIP_CUDA::color3::XYZToYCxCz(FLIP_CUDA::color3::LinearRGBToXYZ(rgb));
        dst[idx] = Vec3<T>(ycxcz.x, ycxcz.y, ycxcz.z);
    }
}

template <typename T>
__global__ void spatial_filter_first_dir_kernel(
    Vec2<T>* __restrict__ inter_ref_ycx,
    Vec2<T>* __restrict__ inter_ref_cz,
    Vec2<T>* __restrict__ inter_test_ycx,
    Vec2<T>* __restrict__ inter_test_cz,
    const Vec3<T>* __restrict__ ref_ycxcz,
    const Vec3<T>* __restrict__ test_ycxcz,
    const FLIP_CUDA::color3* __restrict__ spatial_filter_ycx,
    const FLIP_CUDA::color3* __restrict__ spatial_filter_cz,
    int width,
    int height,
    int filterRadius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int dst_idx = y * width + x;

    float r_y = 0.0f, r_cx = 0.0f, r_cz1 = 0.0f, r_cz2 = 0.0f;
    float t_y = 0.0f, t_cx = 0.0f, t_cz1 = 0.0f, t_cz2 = 0.0f;

    for (int ix = -filterRadius; ix <= filterRadius; ++ix) {
        int xx = min(max(0, x + ix), width - 1);
        int filter_idx = ix + filterRadius;
        int src_idx = y * width + xx;

        FLIP_CUDA::color3 w_ycx = spatial_filter_ycx[filter_idx];
        FLIP_CUDA::color3 w_cz = spatial_filter_cz[filter_idx];

        Vec3<T> r_val = ref_ycxcz[src_idx];
        Vec3<T> t_val = test_ycxcz[src_idx];

        float rx = RealTraits<T>::to_float(r_val.x);
        float ry = RealTraits<T>::to_float(r_val.y);
        float rz = RealTraits<T>::to_float(r_val.z);

        float tx = RealTraits<T>::to_float(t_val.x);
        float ty = RealTraits<T>::to_float(t_val.y);
        float tz = RealTraits<T>::to_float(t_val.z);

        r_y += w_ycx.x * rx;
        r_cx += w_ycx.y * ry;
        r_cz1 += w_cz.x * rz;
        r_cz2 += w_cz.y * rz;

        t_y += w_ycx.x * tx;
        t_cx += w_ycx.y * ty;
        t_cz1 += w_cz.x * tz;
        t_cz2 += w_cz.y * tz;
    }

    inter_ref_ycx[dst_idx] = Vec2<T>(r_y, r_cx);
    inter_ref_cz[dst_idx] = Vec2<T>(r_cz1, r_cz2);
    inter_test_ycx[dst_idx] = Vec2<T>(t_y, t_cx);
    inter_test_cz[dst_idx] = Vec2<T>(t_cz1, t_cz2);
}

template <typename T>
__global__ void spatial_filter_second_dir_and_color_difference_kernel(
    T* __restrict__ color_diff,
    const Vec2<T>* __restrict__ inter_ref_ycx,
    const Vec2<T>* __restrict__ inter_ref_cz,
    const Vec2<T>* __restrict__ inter_test_ycx,
    const Vec2<T>* __restrict__ inter_test_cz,
    const FLIP_CUDA::color3* __restrict__ spatial_filter_ycx,
    const FLIP_CUDA::color3* __restrict__ spatial_filter_cz,
    int width,
    int height,
    int filterRadius,
    float cmax,
    float pccmax
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int dst_idx = y * width + x;

    float r_y = 0.0f, r_cx = 0.0f, r_cz1 = 0.0f, r_cz2 = 0.0f;
    float t_y = 0.0f, t_cx = 0.0f, t_cz1 = 0.0f, t_cz2 = 0.0f;

    for (int iy = -filterRadius; iy <= filterRadius; ++iy) {
        int yy = min(max(0, y + iy), height - 1);
        int filter_idx = iy + filterRadius;
        int src_idx = yy * width + x;

        FLIP_CUDA::color3 w_ycx = spatial_filter_ycx[filter_idx];
        FLIP_CUDA::color3 w_cz = spatial_filter_cz[filter_idx];

        Vec2<T> r_ycx = inter_ref_ycx[src_idx];
        Vec2<T> r_cz = inter_ref_cz[src_idx];
        Vec2<T> t_ycx = inter_test_ycx[src_idx];
        Vec2<T> t_cz = inter_test_cz[src_idx];

        float rx = RealTraits<T>::to_float(r_ycx.x);
        float ry = RealTraits<T>::to_float(r_ycx.y);
        float rz1 = RealTraits<T>::to_float(r_cz.x);
        float rz2 = RealTraits<T>::to_float(r_cz.y);

        float tx = RealTraits<T>::to_float(t_ycx.x);
        float ty = RealTraits<T>::to_float(t_ycx.y);
        float tz1 = RealTraits<T>::to_float(t_cz.x);
        float tz2 = RealTraits<T>::to_float(t_cz.y);

        r_y += w_ycx.x * rx;
        r_cx += w_ycx.y * ry;
        r_cz1 += w_cz.x * rz1;
        r_cz2 += w_cz.y * rz2;

        t_y += w_ycx.x * tx;
        t_cx += w_ycx.y * ty;
        t_cz1 += w_cz.x * tz1;
        t_cz2 += w_cz.y * tz2;
    }

    FLIP_CUDA::color3 r_filt_ycxcz(r_y, r_cx, r_cz1 + r_cz2);
    FLIP_CUDA::color3 t_filt_ycxcz(t_y, t_cx, t_cz1 + t_cz2);

    FLIP_CUDA::color3 r_rgb = FLIP_CUDA::color3::clamp(FLIP_CUDA::color3::XYZToLinearRGB(FLIP_CUDA::color3::YCxCzToXYZ(r_filt_ycxcz)));
    FLIP_CUDA::color3 t_rgb = FLIP_CUDA::color3::clamp(FLIP_CUDA::color3::XYZToLinearRGB(FLIP_CUDA::color3::YCxCzToXYZ(t_filt_ycxcz)));

    FLIP_CUDA::color3 r_lab = FLIP_CUDA::color3::XYZToCIELab(FLIP_CUDA::color3::LinearRGBToXYZ(r_rgb));
    FLIP_CUDA::color3 t_lab = FLIP_CUDA::color3::XYZToCIELab(FLIP_CUDA::color3::LinearRGBToXYZ(t_rgb));

    r_lab.y = FLIP_CUDA::color3::Hunt(r_lab.x, r_lab.y);
    r_lab.z = FLIP_CUDA::color3::Hunt(r_lab.x, r_lab.z);
    t_lab.y = FLIP_CUDA::color3::Hunt(t_lab.x, t_lab.y);
    t_lab.z = FLIP_CUDA::color3::Hunt(t_lab.x, t_lab.z);

    float deltaE = FLIP_CUDA::color3::HyAB(r_lab, t_lab);
    deltaE = powf(deltaE, FLIP_CUDA::DeviceFLIPConstants.gqc);

    if (deltaE < pccmax) {
        deltaE *= FLIP_CUDA::DeviceFLIPConstants.gpt / pccmax;
    } else {
        deltaE = FLIP_CUDA::DeviceFLIPConstants.gpt + ((deltaE - pccmax) / (cmax - pccmax)) * (1.0f - FLIP_CUDA::DeviceFLIPConstants.gpt);
    }

    color_diff[dst_idx] = RealTraits<T>::from_float(deltaE);
}

template <typename T>
__global__ void feature_filter_first_dir_kernel(
    Vec3<T>* __restrict__ inter_feat_ref,
    Vec3<T>* __restrict__ inter_feat_test,
    const Vec3<T>* __restrict__ ref_ycxcz,
    const Vec3<T>* __restrict__ test_ycxcz,
    const FLIP_CUDA::color3* __restrict__ feature_filter,
    int width,
    int height,
    int halfFilterWidth
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int dst_idx = y * width + x;

    float dx_ref = 0.0f, dx_test = 0.0f;
    float ddx_ref = 0.0f, ddx_test = 0.0f;
    float g_ref = 0.0f, g_test = 0.0f;

    for (int ix = -halfFilterWidth; ix <= halfFilterWidth; ++ix) {
        int xx = min(max(0, x + ix), width - 1);
        int filter_idx = ix + halfFilterWidth;
        int src_idx = y * width + xx;

        FLIP_CUDA::color3 fw = feature_filter[filter_idx];
        float y_ref = RealTraits<T>::to_float(ref_ycxcz[src_idx].x);
        float y_test = RealTraits<T>::to_float(test_ycxcz[src_idx].x);

        float y_ref_norm = (y_ref + 16.0f) / 116.0f;
        float y_test_norm = (y_test + 16.0f) / 116.0f;

        dx_ref += fw.y * y_ref_norm;
        dx_test += fw.y * y_test_norm;

        ddx_ref += fw.z * y_ref_norm;
        ddx_test += fw.z * y_test_norm;

        g_ref += fw.x * y_ref_norm;
        g_test += fw.x * y_test_norm;
    }

    inter_feat_ref[dst_idx] = Vec3<T>(dx_ref, ddx_ref, g_ref);
    inter_feat_test[dst_idx] = Vec3<T>(dx_test, ddx_test, g_test);
}

template <typename T>
__global__ void feature_filter_second_dir_and_feature_difference_kernel(
    T* __restrict__ feature_diff,
    const Vec3<T>* __restrict__ inter_feat_ref,
    const Vec3<T>* __restrict__ inter_feat_test,
    const FLIP_CUDA::color3* __restrict__ feature_filter,
    int width,
    int height,
    int halfFilterWidth
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int dst_idx = y * width + x;

    const float normalizationFactor = 1.0f / sqrtf(2.0f);

    float dx_ref = 0.0f, dx_test = 0.0f;
    float ddx_ref = 0.0f, ddx_test = 0.0f;
    float dy_ref = 0.0f, dy_test = 0.0f;
    float ddy_ref = 0.0f, ddy_test = 0.0f;

    for (int iy = -halfFilterWidth; iy <= halfFilterWidth; ++iy) {
        int yy = min(max(0, y + iy), height - 1);
        int filter_idx = iy + halfFilterWidth;
        int src_idx = yy * width + x;

        FLIP_CUDA::color3 fw = feature_filter[filter_idx];
        Vec3<T> feat_ref = inter_feat_ref[src_idx];
        Vec3<T> feat_test = inter_feat_test[src_idx];

        float r_x = RealTraits<T>::to_float(feat_ref.x); float r_y = RealTraits<T>::to_float(feat_ref.y); float r_z = RealTraits<T>::to_float(feat_ref.z);
        float t_x = RealTraits<T>::to_float(feat_test.x); float t_y = RealTraits<T>::to_float(feat_test.y); float t_z = RealTraits<T>::to_float(feat_test.z);

        dx_ref += fw.x * r_x;
        dx_test += fw.x * t_x;
        ddx_ref += fw.x * r_y;
        ddx_test += fw.x * t_y;

        dy_ref += fw.y * r_z;
        dy_test += fw.y * t_z;
        ddy_ref += fw.z * r_z;
        ddy_test += fw.z * t_z;
    }

    float edgeValueRef = sqrtf(dx_ref * dx_ref + dy_ref * dy_ref);
    float edgeValueTest = sqrtf(dx_test * dx_test + dy_test * dy_test);
    float pointValueRef = sqrtf(ddx_ref * ddx_ref + ddy_ref * ddy_ref);
    float pointValueTest = sqrtf(ddx_test * ddx_test + ddy_test * ddy_test);
    float edgeDifference = fabsf(edgeValueRef - edgeValueTest);
    float pointDifference = fabsf(pointValueRef - pointValueTest);
    float featureDifference = powf(normalizationFactor * fmaxf(edgeDifference, pointDifference), 0.5f);

    feature_diff[dst_idx] = RealTraits<T>::from_float(featureDifference);
}

template <typename T>
__global__ void final_error_kernel(
    float* __restrict__ d_out,
    const T* __restrict__ color_diff,
    const T* __restrict__ feature_diff,
    int total_pixels
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_pixels) return;

    float cdiff = RealTraits<T>::to_float(color_diff[i]);
    float fdiff = RealTraits<T>::to_float(feature_diff[i]);
    d_out[i] = powf(cdiff, 1.0f - fdiff);
}

template <typename T>
static bool flip_evaluate_fast_cuda(
    const float* ref_interleaved,
    const float* test_interleaved,
    const VulkanPlaneDesc* ref_vulkan,
    const VulkanPlaneDesc* test_vulkan,
    const uint8_t* const ref_host_ptrs[3],
    const ptrdiff_t ref_host_strides[3],
    const uint8_t* const test_host_ptrs[3],
    const ptrdiff_t test_host_strides[3],
    int width,
    int height,
    const FlipEvaluationParams& params,
    int device_id,
    float* output_data_host,
    uint8_t* const out_ptrs[3],
    const ptrdiff_t out_strides[3],
    double& mean_error
) {
    size_t num_pixels = static_cast<size_t>(width) * height;

    float ppd = 67.0f;
    if (params.use_vc) {
        ppd = FLIP_CUDA::calculatePPD(params.vc[0], params.vc[1], params.vc[2]);
    } else if (params.ppd > 0.0f) {
        ppd = params.ppd;
    }

    auto ws = s_workspace_pool.acquire(num_pixels);
    if (!ws->ensure_filters(ppd)) {
        s_workspace_pool.release(std::move(ws));
        return false;
    }

    auto align256 = [](size_t s) -> size_t {
        return (s + 255) & ~static_cast<size_t>(255);
    };

    size_t sz_v3 = align256(num_pixels * sizeof(Vec3<T>));
    size_t sz_v2 = align256(num_pixels * sizeof(Vec2<T>));
    size_t sz_t  = align256(num_pixels * sizeof(T));
    size_t sz_f  = align256(num_pixels * sizeof(float));
    size_t sz_sums = align256(sizeof(double));
    size_t sz_plane = align256(num_pixels * sizeof(float));

    // Shared scratch area for mutually exclusive execution phases:
    // 1. Host upload: 6 planes of float (24 B/px)
    // 2. Spatial filtering intermediate: 4 planes of Vec2<T> (16 B/px)
    // 3. Feature filtering intermediate: 2 planes of Vec3<T> (12 B/px)
    // 4. Magma host output: 3 planes of float (12 B/px)
    size_t sz_host_upload = (ref_host_ptrs ? 3 * sz_plane : 0) + (test_host_ptrs ? 3 * sz_plane : 0);
    size_t sz_scratch = std::max(sz_host_upload, std::max(sz_v2 * 4, sz_v3 * 2));

    size_t total_required = sz_v3 * 2 + sz_t * 2 + sz_f + sz_sums + sz_scratch;

    if (!ws->ensure_size(total_required)) {
        s_workspace_pool.release(std::move(ws));
        return false;
    }

    uint8_t* ptr = static_cast<uint8_t*>(ws->d_workspace);
    Vec3<T>* d_ref_ycxcz = reinterpret_cast<Vec3<T>*>(ptr); ptr += sz_v3;
    Vec3<T>* d_test_ycxcz = reinterpret_cast<Vec3<T>*>(ptr); ptr += sz_v3;
    T* d_color_diff = reinterpret_cast<T*>(ptr); ptr += sz_t;
    T* d_feature_diff = reinterpret_cast<T*>(ptr); ptr += sz_t;
    float* d_error_map = reinterpret_cast<float*>(ptr); ptr += sz_f;
    double* d_block_sum = reinterpret_cast<double*>(ptr); ptr += sz_sums;

    // Scratch area pointers
    uint8_t* scratch_ptr = ptr;
    float* d_host_ref_planes[3] = { nullptr, nullptr, nullptr };
    float* d_host_test_planes[3] = { nullptr, nullptr, nullptr };
    if (ref_host_ptrs) {
        for (int p = 0; p < 3; ++p) {
            d_host_ref_planes[p] = reinterpret_cast<float*>(scratch_ptr); scratch_ptr += sz_plane;
        }
    }
    if (test_host_ptrs) {
        for (int p = 0; p < 3; ++p) {
            d_host_test_planes[p] = reinterpret_cast<float*>(scratch_ptr); scratch_ptr += sz_plane;
        }
    }

    // Spatial intermediate buffers reuse scratch (valid because host upload is already converted to ycxcz)
    Vec2<T>* d_inter_ref_ycx = reinterpret_cast<Vec2<T>*>(ptr);
    Vec2<T>* d_inter_ref_cz = reinterpret_cast<Vec2<T>*>(ptr + sz_v2);
    Vec2<T>* d_inter_test_ycx = reinterpret_cast<Vec2<T>*>(ptr + sz_v2 * 2);
    Vec2<T>* d_inter_test_cz = reinterpret_cast<Vec2<T>*>(ptr + sz_v2 * 3);

    // Feature intermediate buffers reuse scratch (valid because spatial filtering is finished)
    Vec3<T>* d_inter_feat_ref = reinterpret_cast<Vec3<T>*>(ptr);
    Vec3<T>* d_inter_feat_test = reinterpret_cast<Vec3<T>*>(ptr + sz_v3);

    dim3 block2d(32, 8);
    dim3 grid2d((width + block2d.x - 1) / block2d.x, (height + block2d.y - 1) / block2d.y);
    cudaStream_t stream = ws->stream;

    if (ref_vulkan && ref_vulkan[0].vkSemaphore != 0 && ref_vulkan[0].readyValue > 0) {
        cudaExternalSemaphore_t refSem = get_or_import_vulkan_semaphore(
            ref_vulkan[0].vkSemaphore, ref_vulkan[0].semHandleType, ref_vulkan[0].semHandle
        );
        if (refSem) {
            cudaExternalSemaphoreWaitParams waitParams = {};
            waitParams.params.fence.value = ref_vulkan[0].readyValue;
            cudaWaitExternalSemaphoresAsync(&refSem, &waitParams, 1, stream);
        }
    }
    if (test_vulkan && test_vulkan[0].vkSemaphore != 0 && test_vulkan[0].readyValue > 0) {
        cudaExternalSemaphore_t testSem = get_or_import_vulkan_semaphore(
            test_vulkan[0].vkSemaphore, test_vulkan[0].semHandleType, test_vulkan[0].semHandle
        );
        if (testSem) {
            cudaExternalSemaphoreWaitParams waitParams = {};
            waitParams.params.fence.value = test_vulkan[0].readyValue;
            cudaWaitExternalSemaphoresAsync(&testSem, &waitParams, 1, stream);
        }
    }

    const float* ref_ptrs[3] = { nullptr, nullptr, nullptr };
    int ref_stride_elements = 0;
    if (ref_vulkan) {
        for (int p = 0; p < 3; ++p) {
            ref_ptrs[p] = static_cast<const float*>(get_or_import_vulkan_plane(ref_vulkan[p]));
            if (!ref_ptrs[p]) {
                s_workspace_pool.release(std::move(ws));
                return false;
            }
        }
        ref_stride_elements = ref_vulkan[0].stride_bytes / sizeof(float);
    } else if (ref_host_ptrs) {
        for (int p = 0; p < 3; ++p) {
            cudaMemcpy2DAsync(
                d_host_ref_planes[p],
                width * sizeof(float),
                ref_host_ptrs[p],
                ref_host_strides[p],
                width * sizeof(float),
                height,
                cudaMemcpyHostToDevice,
                stream
            );
            ref_ptrs[p] = d_host_ref_planes[p];
        }
        ref_stride_elements = width;
    }

    const float* test_ptrs[3] = { nullptr, nullptr, nullptr };
    int test_stride_elements = 0;
    if (test_vulkan) {
        for (int p = 0; p < 3; ++p) {
            test_ptrs[p] = static_cast<const float*>(get_or_import_vulkan_plane(test_vulkan[p]));
            if (!test_ptrs[p]) {
                s_workspace_pool.release(std::move(ws));
                return false;
            }
        }
        test_stride_elements = test_vulkan[0].stride_bytes / sizeof(float);
    } else if (test_host_ptrs) {
        for (int p = 0; p < 3; ++p) {
            cudaMemcpy2DAsync(
                d_host_test_planes[p],
                width * sizeof(float),
                test_host_ptrs[p],
                test_host_strides[p],
                width * sizeof(float),
                height,
                cudaMemcpyHostToDevice,
                stream
            );
            test_ptrs[p] = d_host_test_planes[p];
        }
        test_stride_elements = width;
    }

    if (ref_ptrs[0] && test_ptrs[0]) {
        pack_and_convert_planar_pair_to_ycxcz_kernel<T><<<grid2d, block2d, 0, stream>>>(
            ref_ptrs[0], ref_ptrs[1], ref_ptrs[2],
            ref_stride_elements, d_ref_ycxcz,
            test_ptrs[0], test_ptrs[1], test_ptrs[2],
            test_stride_elements, d_test_ycxcz,
            width, height
        );
    } else if (ref_interleaved && test_interleaved) {
        float* d_temp_ref = nullptr;
        float* d_temp_test = nullptr;
        cudaMallocAsync(&d_temp_ref, num_pixels * 3 * sizeof(float), stream);
        cudaMallocAsync(&d_temp_test, num_pixels * 3 * sizeof(float), stream);
        cudaMemcpyAsync(d_temp_ref, ref_interleaved, num_pixels * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_temp_test, test_interleaved, num_pixels * 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

        convert_interleaved_to_ycxcz_kernel<T><<<grid2d, block2d, 0, stream>>>(d_temp_ref, d_ref_ycxcz, width, height);
        convert_interleaved_to_ycxcz_kernel<T><<<grid2d, block2d, 0, stream>>>(d_temp_test, d_test_ycxcz, width, height);

        cudaFreeAsync(d_temp_ref, stream);
        cudaFreeAsync(d_temp_test, stream);
    } else {
        s_workspace_pool.release(std::move(ws));
        return false;
    }

    const float cmax = FLIP_CUDA::color3::computeMaxDistance(FLIP_CUDA::FLIPConstants.gqc);
    const float pccmax = FLIP_CUDA::FLIPConstants.gpc * cmax;

    spatial_filter_first_dir_kernel<T><<<grid2d, block2d, 0, stream>>>(
        d_inter_ref_ycx,
        d_inter_ref_cz,
        d_inter_test_ycx,
        d_inter_test_cz,
        d_ref_ycxcz,
        d_test_ycxcz,
        ws->spatialFilterYCx->getDeviceData(),
        ws->spatialFilterCz->getDeviceData(),
        width,
        height,
        ws->spatial_filter_radius
    );

    spatial_filter_second_dir_and_color_difference_kernel<T><<<grid2d, block2d, 0, stream>>>(
        d_color_diff,
        d_inter_ref_ycx,
        d_inter_ref_cz,
        d_inter_test_ycx,
        d_inter_test_cz,
        ws->spatialFilterYCx->getDeviceData(),
        ws->spatialFilterCz->getDeviceData(),
        width,
        height,
        ws->spatial_filter_radius,
        cmax,
        pccmax
    );

    feature_filter_first_dir_kernel<T><<<grid2d, block2d, 0, stream>>>(
        d_inter_feat_ref,
        d_inter_feat_test,
        d_ref_ycxcz,
        d_test_ycxcz,
        ws->featureFilter->getDeviceData(),
        width,
        height,
        ws->feature_filter_radius
    );

    feature_filter_second_dir_and_feature_difference_kernel<T><<<grid2d, block2d, 0, stream>>>(
        d_feature_diff,
        d_inter_feat_ref,
        d_inter_feat_test,
        ws->featureFilter->getDeviceData(),
        width,
        height,
        ws->feature_filter_radius
    );

    int blockSize1d = 256;
    int numBlocks1d = (static_cast<int>(num_pixels) + blockSize1d - 1) / blockSize1d;
    final_error_kernel<T><<<numBlocks1d, blockSize1d, 0, stream>>>(d_error_map, d_color_diff, d_feature_diff, static_cast<int>(num_pixels));

    const unsigned int redBlockSize = 256;
    unsigned int redNumBlocks = static_cast<unsigned int>((num_pixels + (redBlockSize * 2 - 1)) / (redBlockSize * 2));
    if (redNumBlocks > 1024) redNumBlocks = 1024;

    cudaMemsetAsync(d_block_sum, 0, sizeof(double), stream);
    fast_reduce_sum_kernel<redBlockSize><<<redNumBlocks, redBlockSize, redBlockSize * sizeof(double), stream>>>(d_error_map, d_block_sum, num_pixels);
    cudaMemcpyAsync(ws->h_pinned_sum, d_block_sum, sizeof(double), cudaMemcpyDeviceToHost, stream);

    if (params.apply_magma) {
        ensure_magma_lut_initialized(device_id);

        if (params.gpu_output && out_ptrs && out_ptrs[0]) {
            magma_map_planar_kernel<<<grid2d, block2d, 0, stream>>>(
                d_error_map,
                reinterpret_cast<float*>(out_ptrs[0]),
                static_cast<int>(out_strides[0] / sizeof(float)),
                reinterpret_cast<float*>(out_ptrs[1]),
                static_cast<int>(out_strides[1] / sizeof(float)),
                reinterpret_cast<float*>(out_ptrs[2]),
                static_cast<int>(out_strides[2] / sizeof(float)),
                width,
                height
            );
        } else if (out_ptrs && out_ptrs[0]) {
            float* d_magma_r = reinterpret_cast<float*>(ptr);
            float* d_magma_g = reinterpret_cast<float*>(ptr + sz_plane);
            float* d_magma_b = reinterpret_cast<float*>(ptr + sz_plane * 2);

            magma_map_planar_kernel<<<grid2d, block2d, 0, stream>>>(
                d_error_map,
                d_magma_r, width,
                d_magma_g, width,
                d_magma_b, width,
                width, height
            );

            cudaError_t e0 = cudaMemcpy2DAsync(out_ptrs[0], out_strides[0], d_magma_r, width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost, stream);
            cudaError_t e1 = cudaMemcpy2DAsync(out_ptrs[1], out_strides[1], d_magma_g, width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost, stream);
            cudaError_t e2 = cudaMemcpy2DAsync(out_ptrs[2], out_strides[2], d_magma_b, width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost, stream);
            if (e0 != cudaSuccess || e1 != cudaSuccess || e2 != cudaSuccess) {
                s_workspace_pool.release(std::move(ws));
                return false;
            }
        } else if (output_data_host) {
            float* d_magma_inter = reinterpret_cast<float*>(ws->d_workspace);
            magma_map_interleaved_kernel<<<grid2d, block2d, 0, stream>>>(
                d_error_map,
                d_magma_inter,
                width, height
            );
            cudaMemcpyAsync(output_data_host, d_magma_inter, num_pixels * sizeof(float) * 3, cudaMemcpyDeviceToHost, stream);
        }
    } else {
        if (params.gpu_output) {
            if (out_ptrs && out_ptrs[0]) {
                cudaError_t copyErr = cudaMemcpy2DAsync(
                    out_ptrs[0],
                    out_strides[0],
                    d_error_map,
                    width * sizeof(float),
                    width * sizeof(float),
                    height,
                    cudaMemcpyDeviceToDevice,
                    stream
                );
                if (copyErr != cudaSuccess) {
                    s_workspace_pool.release(std::move(ws));
                    return false;
                }
            }
        } else {
            if (out_ptrs && out_ptrs[0]) {
                cudaError_t copyErr = cudaMemcpy2DAsync(
                    out_ptrs[0],
                    out_strides[0],
                    d_error_map,
                    width * sizeof(float),
                    width * sizeof(float),
                    height,
                    cudaMemcpyDeviceToHost,
                    stream
                );
                if (copyErr != cudaSuccess) {
                    s_workspace_pool.release(std::move(ws));
                    return false;
                }
            } else if (output_data_host) {
                cudaMemcpyAsync(output_data_host, d_error_map, num_pixels * sizeof(float), cudaMemcpyDeviceToHost, stream);
            }
        }
    }

    cudaStreamSynchronize(stream);
    mean_error = *(ws->h_pinned_sum) / static_cast<double>(num_pixels);
    s_workspace_pool.release(std::move(ws));
    return true;
}

bool flip_cuda_is_available() {
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
}

int flip_cuda_device_count() {
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess) ? count : 0;
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
    if (!ref_interleaved || !test_interleaved || !output_data || width <= 0 || height <= 0) return false;

    int dev_count = flip_cuda_device_count();
    if (dev_count <= 0) return false;
    if (device_id < 0 || device_id >= dev_count) device_id = 0;
    if (cudaSetDevice(device_id) != cudaSuccess) return false;

    // optin fast path (fp16) for ldr
    if (params.dynamic_range == 0 && params.fast) {
        return flip_evaluate_fast_cuda<__half>(
            ref_interleaved, test_interleaved, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr,
            width, height, params, device_id, output_data, nullptr, nullptr, mean_error
        );
    }

    // default fp32 baseline
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

        FLIP_CUDA::image<FLIP_CUDA::color3> referenceImage;
        FLIP_CUDA::image<FLIP_CUDA::color3> testImage;
        FLIP_CUDA::image<float> errorMapFLIPOutputImage(width, height, 0.0f);

        referenceImage.setPixels(ref_interleaved, width, height);
        testImage.setPixels(test_interleaved, width, height);

        FLIP_CUDA::evaluate(referenceImage, testImage, use_hdr, flip_params, errorMapFLIPOutputImage);

        mean_error = compute_gpu_mean(errorMapFLIPOutputImage.getDeviceData(), num_pixels);

        if (params.apply_magma) {
            ensure_magma_lut_initialized(device_id);
            dim3 block2d(32, 8);
            dim3 grid2d((width + block2d.x - 1) / block2d.x, (height + block2d.y - 1) / block2d.y);
            float* d_inter = reinterpret_cast<float*>(referenceImage.getDeviceData());
            magma_map_interleaved_kernel<<<grid2d, block2d>>>(
                errorMapFLIPOutputImage.getDeviceData(),
                d_inter,
                width,
                height
            );
            cudaMemcpy(output_data, d_inter, num_pixels * sizeof(float) * 3, cudaMemcpyDeviceToHost);
        } else {
            errorMapFLIPOutputImage.synchronizeHost();
            std::memcpy(output_data, errorMapFLIPOutputImage.getHostData(), num_pixels * sizeof(float));
        }

        cudaDeviceSynchronize();
        return true;
    } catch (...) {
        return false;
    }
}

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
) {
    if (width <= 0 || height <= 0 || !out_ptrs[0]) return false;
    if (!ref_vulkan && (!ref_host_ptrs || !ref_host_ptrs[0])) return false;
    if (!test_vulkan && (!test_host_ptrs || !test_host_ptrs[0])) return false;

    int dev_count = flip_cuda_device_count();
    if (dev_count <= 0) return false;
    if (device_id < 0 || device_id >= dev_count) device_id = 0;
    if (cudaSetDevice(device_id) != cudaSuccess) return false;

    // fast fp16 path
    if (params.dynamic_range == 0 && params.fast) {
        return flip_evaluate_fast_cuda<__half>(
            nullptr, nullptr, ref_vulkan, test_vulkan,
            ref_host_ptrs, ref_host_strides,
            test_host_ptrs, test_host_strides,
            width, height, params, device_id,
            nullptr, out_ptrs, out_strides, mean_error
        );
    }

    // default fp32 baseline
    try {
        size_t num_pixels = static_cast<size_t>(width) * height;

        if (ref_vulkan && ref_vulkan[0].vkSemaphore != 0 && ref_vulkan[0].readyValue > 0) {
            cudaExternalSemaphore_t refSem = get_or_import_vulkan_semaphore(
                ref_vulkan[0].vkSemaphore, ref_vulkan[0].semHandleType, ref_vulkan[0].semHandle
            );
            if (refSem) {
                cudaExternalSemaphoreWaitParams waitParams = {};
                waitParams.params.fence.value = ref_vulkan[0].readyValue;
                cudaWaitExternalSemaphoresAsync(&refSem, &waitParams, 1, 0);
            }
        }
        if (test_vulkan && test_vulkan[0].vkSemaphore != 0 && test_vulkan[0].readyValue > 0) {
            cudaExternalSemaphore_t testSem = get_or_import_vulkan_semaphore(
                test_vulkan[0].vkSemaphore, test_vulkan[0].semHandleType, test_vulkan[0].semHandle
            );
            if (testSem) {
                cudaExternalSemaphoreWaitParams waitParams = {};
                waitParams.params.fence.value = test_vulkan[0].readyValue;
                cudaWaitExternalSemaphoresAsync(&testSem, &waitParams, 1, 0);
            }
        }

        const float* ref_ptrs[3] = {};
        float *d_r_ref = nullptr, *d_g_ref = nullptr, *d_b_ref = nullptr;
        int ref_stride_elements = 0;
        if (ref_vulkan) {
            for (int p = 0; p < 3; ++p) {
                ref_ptrs[p] = static_cast<const float*>(get_or_import_vulkan_plane(ref_vulkan[p]));
                if (!ref_ptrs[p]) return false;
            }
            ref_stride_elements = ref_vulkan[0].stride_bytes / sizeof(float);
        } else {
            cudaMalloc(&d_r_ref, num_pixels * sizeof(float));
            cudaMalloc(&d_g_ref, num_pixels * sizeof(float));
            cudaMalloc(&d_b_ref, num_pixels * sizeof(float));
            cudaMemcpy2D(d_r_ref, width * sizeof(float), ref_host_ptrs[0], ref_host_strides[0], width * sizeof(float), height, cudaMemcpyHostToDevice);
            cudaMemcpy2D(d_g_ref, width * sizeof(float), ref_host_ptrs[1], ref_host_strides[1], width * sizeof(float), height, cudaMemcpyHostToDevice);
            cudaMemcpy2D(d_b_ref, width * sizeof(float), ref_host_ptrs[2], ref_host_strides[2], width * sizeof(float), height, cudaMemcpyHostToDevice);
            ref_ptrs[0] = d_r_ref; ref_ptrs[1] = d_g_ref; ref_ptrs[2] = d_b_ref;
            ref_stride_elements = width;
        }

        const float* test_ptrs[3] = {};
        float *d_r_test = nullptr, *d_g_test = nullptr, *d_b_test = nullptr;
        int test_stride_elements = 0;
        if (test_vulkan) {
            for (int p = 0; p < 3; ++p) {
                test_ptrs[p] = static_cast<const float*>(get_or_import_vulkan_plane(test_vulkan[p]));
                if (!test_ptrs[p]) {
                    if (d_r_ref) { cudaFree(d_r_ref); cudaFree(d_g_ref); cudaFree(d_b_ref); }
                    return false;
                }
            }
            test_stride_elements = test_vulkan[0].stride_bytes / sizeof(float);
        } else {
            cudaMalloc(&d_r_test, num_pixels * sizeof(float));
            cudaMalloc(&d_g_test, num_pixels * sizeof(float));
            cudaMalloc(&d_b_test, num_pixels * sizeof(float));
            cudaMemcpy2D(d_r_test, width * sizeof(float), test_host_ptrs[0], test_host_strides[0], width * sizeof(float), height, cudaMemcpyHostToDevice);
            cudaMemcpy2D(d_g_test, width * sizeof(float), test_host_ptrs[1], test_host_strides[1], width * sizeof(float), height, cudaMemcpyHostToDevice);
            cudaMemcpy2D(d_b_test, width * sizeof(float), test_host_ptrs[2], test_host_strides[2], width * sizeof(float), height, cudaMemcpyHostToDevice);
            test_ptrs[0] = d_r_test; test_ptrs[1] = d_g_test; test_ptrs[2] = d_b_test;
            test_stride_elements = width;
        }

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

        FLIP_CUDA::image<FLIP_CUDA::color3> referenceImage(width, height);
        FLIP_CUDA::image<FLIP_CUDA::color3> testImage(width, height);
        FLIP_CUDA::image<float> errorMapFLIPOutputImage(width, height, 0.0f);

        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        pack_planar_to_color3_kernel<<<gridDim, blockDim>>>(
            ref_ptrs[0], ref_ptrs[1], ref_ptrs[2],
            ref_stride_elements, referenceImage.getDeviceData(), width, height
        );
        referenceImage.setState(FLIP_CUDA::CudaTensorState::DEVICE_ONLY);

        pack_planar_to_color3_kernel<<<gridDim, blockDim>>>(
            test_ptrs[0], test_ptrs[1], test_ptrs[2],
            test_stride_elements, testImage.getDeviceData(), width, height
        );
        testImage.setState(FLIP_CUDA::CudaTensorState::DEVICE_ONLY);

        cudaDeviceSynchronize();
        if (d_r_ref) { cudaFree(d_r_ref); cudaFree(d_g_ref); cudaFree(d_b_ref); }
        if (d_r_test) { cudaFree(d_r_test); cudaFree(d_g_test); cudaFree(d_b_test); }

        FLIP_CUDA::evaluate(referenceImage, testImage, use_hdr, flip_params, errorMapFLIPOutputImage);
        mean_error = compute_gpu_mean(errorMapFLIPOutputImage.getDeviceData(), num_pixels);

        if (params.apply_magma) {
            ensure_magma_lut_initialized(device_id);

            if (params.gpu_output && out_ptrs && out_ptrs[0]) {
                magma_map_planar_kernel<<<gridDim, blockDim>>>(
                    errorMapFLIPOutputImage.getDeviceData(),
                    reinterpret_cast<float*>(out_ptrs[0]),
                    static_cast<int>(out_strides[0] / sizeof(float)),
                    reinterpret_cast<float*>(out_ptrs[1]),
                    static_cast<int>(out_strides[1] / sizeof(float)),
                    reinterpret_cast<float*>(out_ptrs[2]),
                    static_cast<int>(out_strides[2] / sizeof(float)),
                    width,
                    height
                );
            } else if (out_ptrs && out_ptrs[0]) {
                float* d_mr = reinterpret_cast<float*>(referenceImage.getDeviceData());
                float* d_mg = d_mr + num_pixels;
                float* d_mb = d_mg + num_pixels;

                magma_map_planar_kernel<<<gridDim, blockDim>>>(
                    errorMapFLIPOutputImage.getDeviceData(),
                    d_mr, width,
                    d_mg, width,
                    d_mb, width,
                    width, height
                );

                cudaMemcpy2D(out_ptrs[0], out_strides[0], d_mr, width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost);
                cudaMemcpy2D(out_ptrs[1], out_strides[1], d_mg, width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost);
                cudaMemcpy2D(out_ptrs[2], out_strides[2], d_mb, width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost);
            }
        } else {
            if (params.gpu_output) {
                cudaMemcpy2D(out_ptrs[0], out_strides[0], errorMapFLIPOutputImage.getDeviceData(), width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToDevice);
            } else {
                cudaMemcpy2D(out_ptrs[0], out_strides[0], errorMapFLIPOutputImage.getDeviceData(), width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost);
            }
        }

        cudaDeviceSynchronize();
        return true;
    } catch (...) {
        return false;
    }
}

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
) {
    return flip_evaluate_cuda_flexible(
        nullptr, ref_ptrs, ref_strides,
        nullptr, test_ptrs, test_strides,
        width, height, params, device_id,
        out_ptrs, out_strides, mean_error
    );
}

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
) {
    ptrdiff_t dummy_strides[3] = {};
    return flip_evaluate_cuda_flexible(
        ref_planes, nullptr, dummy_strides,
        test_planes, nullptr, dummy_strides,
        width, height, params, device_id,
        out_ptrs, out_strides, mean_error
    );
}

#else

bool flip_cuda_is_available() { return false; }
int flip_cuda_device_count() { return 0; }

bool flip_evaluate_cuda(
    const float*, const float*, int, int,
    const FlipEvaluationParams&, int, float*, double&
) { return false; }

bool flip_evaluate_cuda_planar(
    const uint8_t* const[3], const ptrdiff_t[3],
    const uint8_t* const[3], const ptrdiff_t[3],
    int, int, const FlipEvaluationParams&, int,
    uint8_t* const[3], const ptrdiff_t[3], double&
) { return false; }

bool flip_evaluate_cuda_vulkan(
    const VulkanPlaneDesc[3], const VulkanPlaneDesc[3],
    int, int, const FlipEvaluationParams&, int,
    uint8_t* const[3], const ptrdiff_t[3], double&
) { return false; }

bool flip_evaluate_cuda_flexible(
    const VulkanPlaneDesc*,
    const uint8_t* const[3],
    const ptrdiff_t[3],
    const VulkanPlaneDesc*,
    const uint8_t* const[3],
    const ptrdiff_t[3],
    int, int,
    const FlipEvaluationParams&,
    int,
    uint8_t* const[3],
    const ptrdiff_t[3],
    double&
) { return false; }

void* flip_cuda_import_vulkan_plane(const VulkanPlaneDesc&) { return nullptr; }
void flip_cuda_cleanup_memory_cache() {}

#endif
