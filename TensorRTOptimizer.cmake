# Automates compilation configurations optimizations across the onboard Tensor Core matrix lines
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIRS})

# Enforce strict ahead-of-time (AOT) matrix compilation optimization flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=sm_87 -gencode arch=compute_87,code=sm_87 --use_fast_math --restrict")

# Links dependencies natively across active GPU streaming layers
function(optimize_neural_target target_name)
    target_link_libraries(${target_name} PRIVATE ${CUDA_LIBRARIES} ${TENSORRT_LIBRARIES})
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()
