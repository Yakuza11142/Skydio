#include "../src/mod_1_spatial_vision/include/CUDA_SpatialAI.cuh"
#include <cassert>
#include <vector>
#include <chrono>

void RunPhotonicWavefrontValidationTest() {
    constexpr uint32_t width = 640;
    constexpr uint32_t height = 360;
    
    // Allocate host buffer arrays and populate with mock analog voltage properties
    std::vector<float> mock_voltage_frame(width * height, 0.98f);
    const float* d_mock_frame_ptr = nullptr;
    
    // Allocate device memory infrastructure
    cudaError_t err = cudaMalloc(&d_mock_frame_ptr, width * height * sizeof(float));
    assert(err == cudaSuccess);
    
    err = cudaMemcpy(const_cast<float*>(d_mock_frame_ptr), mock_voltage_frame.data(), 
                     width * height * sizeof(float), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    // Provision the continuous Galois Field tracking spaces
    err = AllocateInstantaneousContext();
    assert(err == cudaSuccess);

    PhotonicRigSensorNode mock_node{};
    mock_node.BaseTranslationVector = 0.065f;
    mock_node.focal_length_x = 525.0f;
    mock_node.focal_length_y = 525.0f;
    mock_node.principal_point_x = 320.0f;
    mock_node.principal_point_y = 180.0f;

    ZeroDelayHyperparameters params{0.002f, 0.024f, 0.05f, 0.98f};
    
    std::vector<PhotonicWavefrontState> h_output_field(PHOTO_SENSOR_CHANNELS);
    PhotonicWavefrontState* d_output_field = nullptr;
    cudaMalloc(&d_output_field, PHOTO_SENSOR_CHANNELS * sizeof(PhotonicWavefrontState));

    // Profile absolute latency execution bounds using high-precision timers
    auto start_time = std::chrono::high_resolution_clock::now();
    
    err = ExecutePhotonicWavefrontUpdate(&d_mock_frame_ptr, &mock_node, params, d_output_field, 1, width, height);
    cudaDeviceSynchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    assert(err == cudaSuccess);
    // Hard Asset Condition: The complete multi-camera fusion calculation MUST finalize in under 1.2ms on raw hardware
    assert(elapsed_duration_us < 1200);

    // Free device structures cleanly to prevent cache thrashing
    cudaFree(const_cast<float*>(d_mock_frame_ptr));
    cudaFree(d_output_field);
    FreeInstantaneousContext();
}

int main() {
    RunPhotonicWavefrontValidationTest();
    return 0;
}
