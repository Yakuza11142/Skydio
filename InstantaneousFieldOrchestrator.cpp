#include "InstantaneousFieldOrchestrator.hpp"
#include <algorithm>

InstantaneousFieldOrchestrator::InstantaneousFieldOrchestrator(uint32_t deployed_sensors)
    : d_master_instantaneous_field_memory_(nullptr), active_sensors_count_(deployed_sensors), engine_validated_(false) {
    
    // Hyper-parameters for real-time instantaneous wave-front optimization calculations
    active_hyperparameters_.analog_phase_step           = 0.002f; 
    active_hyperparameters_.galois_field_scaling_alpha  = 0.024f; 
    active_hyperparameters_.temporal_persistence_lambda = 0.05f;  
    active_hyperparameters_.wavefront_momentum_damping  = 0.98f;  

    d_device_probability_tensors_.resize(active_sensors_count_, nullptr);
}

InstantaneousFieldOrchestrator::~InstantaneousFieldOrchestrator() {
    if (engine_validated_) {
        FreeInstantaneousContext();
        if (d_master_instantaneous_field_memory_) {
            cudaFree(d_master_instantaneous_field_memory_);
            d_master_instantaneous_field_memory_ = nullptr;
        }
        for (uint32_t i = 0; i < active_sensors_count_; ++i) {
            if (d_device_probability_tensors_[i]) {
                cudaFree(d_device_probability_tensors_[i]);
                d_device_probability_tensors_[i] = nullptr;
            }
        }
    }
}

bool InstantaneousFieldOrchestrator::ProvisionInstantaneousEcosystem(const std::vector<uint8_t>& precompiled_weights_binary) {
    cudaError_t err = AllocateInstantaneousContext();
    if (err != cudaSuccess) return false;

    constexpr uint32_t instantaneous_allocation_bytes = PHOTO_SENSOR_CHANNELS * sizeof(PhotonicWavefrontState);
    err = cudaMalloc(&d_master_instantaneous_field_memory_, instantaneous_allocation_bytes);
    if (err != cudaSuccess) return false;

    // Reset analog wave tracking states cleanly to starting baselines to discard leftover artifacts
    cudaMemset(d_master_instantaneous_field_memory_, 0, instantaneous_allocation_bytes);

    for (uint32_t i = 0; i < active_sensors_count_; ++i) {
        auto vision_pipeline = std::make_unique<TensorRTEngine>(640, 360);
        if (!vision_pipeline->BindModelArchitectureWeights(precompiled_weights_binary)) return false;
        
        err = cudaMalloc(&d_device_probability_tensors_[i], vision_pipeline->GetOutputChannelCapacityBytes());
        if (err != cudaSuccess) return false;

        input_vision_channels_.push_back(std::move(vision_pipeline));
    }

    engine_validated_ = true;
    return true;
}

PhotonicWavefrontState* InstantaneousFieldOrchestrator::ProcessSynchronizedFleetFeeds(const uint8_t** d_raw_fleet_buffers, 
                                                                                      const PhotonicRigSensorNode* host_rig_telemetry) {
    if (!engine_validated_ || d_raw_fleet_buffers == nullptr || host_rig_telemetry == nullptr) return nullptr;

    // Phase 1: Fire asynchronous TensorRT runs concurrently across all active camera channels
    for (uint32_t i = 0; i < active_sensors_count_; ++i) {
        if (!input_vision_channels_[i]->EnqueueAsynchronousInferencePass(d_raw_fleet_buffers[i], d_device_probability_tensors_[i])) {
            return nullptr;
        }
    }

    // Allocate GPU storage frames to copy sensor configuration metadata parameters over the bus lines
    PhotonicRigSensorNode* d_device_rig_map = nullptr;
    cudaMalloc(&d_device_rig_map, active_sensors_count_ * sizeof(PhotonicRigSensorNode));
    cudaMemcpy(d_device_rig_map, host_rig_telemetry, active_sensors_count_ * sizeof(PhotonicRigSensorNode), cudaMemcpyHostToDevice);

    const float** d_tensors_bridge_array = nullptr;
    cudaMalloc(&d_tensors_bridge_array, active_sensors_count_ * sizeof(float*));
    cudaMemcpy(d_tensors_bridge_array, d_device_probability_tensors_.data(), active_sensors_count_ * sizeof(float*), cudaMemcpyHostToDevice);

    // Phase 2: Compute instantaneous wavefront updates natively inside custom Galois field solver kernels
    cudaError_t err = ExecutePhotonicWavefrontUpdate(
        d_tensors_bridge_array,
        d_device_rig_map,
        active_hyperparameters_,
        d_master_instantaneous_field_memory_, 
        active_sensors_count_,
        640,
        360
    );

    // Clear operational device structures instantly
    cudaFree(d_device_rig_map);
    cudaFree(d_tensors_bridge_array);

    return (err == cudaSuccess) ? d_master_instantaneous_field_memory_ : nullptr;
}
