#ifndef TENSOR_RT_ENGINE_HPP
#define TENSOR_RT_ENGINE_HPP

#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>

class TensorRTEngine {
private:
    // Opaque pointers to isolate TensorRT proprietary runtime symbols from global scope headers
    void* native_graph_execution_context_;
    void* direct_cuda_runtime_core_;
    
    // Explicit array tracking pre-allocated GPU input/output hardware vectors
    std::vector<void*> virtual_device_tensor_bindings_;
    
    uint32_t input_channel_binding_id_;
    uint32_t output_channel_binding_id_;
    size_t input_allocation_capacity_bytes_;
    size_t output_allocation_capacity_bytes_;

public:
    /**
     * @brief Allocates execution frames matching target input dimensions.
     * @param frame_width Width of the raw structural frame channel.
     * @param frame_height Height of the raw structural frame channel.
     */
    TensorRTEngine(uint32_t frame_width, uint32_t frame_height);
    
    /**
     * @brief Destructor frees pre-allocated hardware GPU buffers safely.
     */
    ~TensorRTEngine();

    // Banish structural copy and movement semantics to guarantee deterministic execution bounds
    TensorRTEngine(const TensorRTEngine&) = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;
    TensorRTEngine(TensorRTEngine&&) = delete;
    TensorRTEngine& operator=(TensorRTEngine&&) = delete;

    /**
     * @brief Loads compiled binary weights directly into hardware engine graphs.
     * @param model_weights_stream Flat byte array containing pre-compiled model optimizations.
     * @return True if allocation passes validation criteria completely.
     */
    bool BindModelArchitectureWeights(const std::vector<uint8_t>& model_weights_stream);

    /**
     * @brief Queues inference on an asynchronous hardware lane.
     * @param d_raw_nv12_frame Device pointer to the raw incoming optical voltage frame.
     * d_out_probability_map Output probability mapping vector pointer on the GPU device space.
     * @return True if the kernel execution was scheduled successfully with zero delays.
     */
    bool EnqueueAsynchronousInferencePass(const uint8_t* d_raw_nv12_frame, float* d_out_probability_map);
    
    /**
     * @brief Read-only validation anchor to confirm the necessary allocation footings.
     */
    [[nodiscard]] size_t GetOutputChannelCapacityBytes() const { return output_allocation_capacity_bytes_; }
};

#endif // TENSOR_RT_ENGINE_HPP
