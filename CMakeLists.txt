cmake_minimum_required(VERSION 3.22)
project(SkydioFleetEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force standard native pthreads lookup to decouple from FindCUDA completely
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Dual-fleet assembly target matrix mapping (Skydio 2, S2+, X10)
set(CMAKE_CUDA_ARCHITECTURES "62;72;87")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC --expt-relaxed-constexpr")

if(CHIP_TARGET STREQUAL "TX2")
    add_compile_definitions(TARGET_HARDWARE_LEGACY_TX2)
    message(STATUS "ENGINE CONFIGURED FOR FLIGHT: SKYDIO 2 (LEGACY TX2)")
else()
    add_compile_definitions(TARGET_HARDWARE_ORIN)
    message(STATUS "ENGINE CONFIGURED FOR FLIGHT: SKYDIO X10 (ORIN PRO)")
endif()

include_directories(src/mod_1_spatial_vision/include)

add_library(skydio_pro_core SHARED
    src/mod_1_spatial_vision/src/CUDA_SpatialAI.cu
)

target_link_libraries(skydio_pro_core PRIVATE Threads::Threads)
