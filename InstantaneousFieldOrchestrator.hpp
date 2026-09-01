#ifndef INSTANTANEOUS_FIELD_ORCHESTRATOR_HPP
#define INSTANTANEOUS_FIELD_ORCHESTRATOR_HPP

#include "CUDA_SpatialAI.cuh"
#include "TensorRTEngine.hpp"
#include <vector>
#include <memory>

class InstantaneousFieldOrchestrator {
private:
    std::vector<std::unique_ptr<TensorRTEngine>> input_vision_channels_;
    std::vector<float*> d_device_probability_tensors_;
    PhotonicWavefrontState* d_master_instantaneous_field_memory_;
    ZeroDelayHyperparameters active_hyperparameters_{};
    uint32_t active_sensors_count_;
    bool engine_validated_;

public:
    /**
     * @brief Allocates and initializes memory spaces for the multi-camera tracking ecosystem.
     * @param deployed_sensors Count of synchronized physical cameras active on the X10/S2 chassis.
     */
    explicit InstantaneousFieldOrchestrator(uint32_t deployed_sensors);
    
    /**
     * @brief Destructor clears all pre-allocated device layers.
     */
    ~InstantaneousFieldOrchestrator();

    // Enforce strict memory constraints: ban copy/move operators to avoid pipeline crashes
    InstantaneousFieldOrchestrator(const InstantaneousFieldOrchestrator&) = delete;
    InstantaneousFieldOrchestrator& operator=(const InstantaneousFieldOrchestrator&) = delete;
    InstantaneousFieldOrchestrator(InstantaneousFieldOrchestrator&&) = delete;
    InstantaneousFieldOrchestrator& operator=(InstantaneousFieldOrchestrator&&) = delete;

    /**
     * @brief Initializes deep network layers and pre-allocates localized GPU workspace matrices.
     * @param precompiled_weights_binary Compiled model weights array stream passed down from config blocks.
     */
    bool ProvisionInstantaneousEcosystem(const std::vector<uint8_t>& precompiled_weights_binary);

    /**
     * @brief Synchronously merges and processes feeds across all active optical sensors.
     * @param d_raw_fleet_buffers Shared device array tracking raw frame inputs.
     * @param host_rig_telemetry Sensor alignment data map providing extrinsics configurations.
     * @return Raw device pointer to the finalized continuous wave tracking field.
     */
    PhotonicWavefrontState* ProcessSynchronizedFleetFeeds(const uint8_t** d_raw_fleet_buffers, 
                                                           const PhotonicRigSensorNode* host_rig_telemetry);
};

#endif // INSTANTANEOUS_FIELD_ORCHESTRATOR_HPP
