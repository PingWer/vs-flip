cmake_minimum_required(VERSION 3.20)
project(vsflip LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(VSFLIP_HAS_CUDA ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Target NVIDIA RTX 2000 series and up (Turing 75, Ampere 80/86, Ada 89, Hopper 90, PTX 89-virtual)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90;89-virtual")
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA enabled with architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    set(VSFLIP_HAS_CUDA OFF)
    message(STATUS "CUDA compiler not found. Building CPU-only.")
endif()

find_package(OpenMP COMPONENTS CXX REQUIRED)
find_package(Vulkan QUIET)

set(SOURCES
    src/vsflip_plugin.cpp
    src/flip_cpu.cpp
)

if(VSFLIP_HAS_CUDA)
    list(APPEND SOURCES src/flip_cuda.cu)
    add_compile_definitions(VSFLIP_CUDA_SUPPORT)
endif()

if(Vulkan_FOUND)
    add_compile_definitions(VSFLIP_HAS_VULKAN)
    message(STATUS "Vulkan found. Vulkan-CUDA interop enabled: ${Vulkan_INCLUDE_DIRS}")
endif()

add_compile_definitions(VS_USE_LATEST_API)

add_library(vsflip SHARED ${SOURCES})

target_include_directories(vsflip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

if(Vulkan_FOUND)
    target_include_directories(vsflip PRIVATE ${Vulkan_INCLUDE_DIRS})
endif()

target_link_libraries(vsflip PRIVATE OpenMP::OpenMP_CXX)

if(VSFLIP_HAS_CUDA)
    target_link_libraries(vsflip PRIVATE CUDA::cudart_static)
endif()

if(WIN32)
    set_target_properties(vsflip PROPERTIES
        PREFIX ""
        OUTPUT_NAME "vsflip"
    )
else()
    set_target_properties(vsflip PROPERTIES
        OUTPUT_NAME "vsflip"
    )
endif()
