#include "CUDA_SpatialAI.cuh"
#include <device_launch_parameters.h>

static float* g_d_photonic_wavefront_scratch_cache = nullptr;
constexpr int PHOTON_BLOCK_X = 32; 
constexpr int PHOTON_BLOCK_Y = 32;

const char* QueryInstantaneousEngineToken(InstantaneousEngineToken token_id) {
    switch (token_id) {
        case InstantaneousEngineToken::ERR_WAVEFRONT_DECOHERENCE:  return "INSTANT_ENGINE_ERR_PHOTONIC_WAVEFRONT_DECOHERENCE";
        case InstantaneousEngineToken::ERR_GALOIS_FIELD_DIVERGE:   return "INSTANT_ENGINE_ERR_GALOIS_FIELD_COMPUTATION_DIVERGENT";
        case InstantaneousEngineToken::ERR_METASURFACE_BREACH:     return "INSTANT_ENGINE_ERR_METASURFACE_HARDWARE_INTERFACE_BREACH";
        case InstantaneousEngineToken::HAL_PHOTONIC_CORE_ENGAGED:  return "INSTANT_ENGINE_HAL_PHOTONIC_FOCAL_PLANE_CORES_ENGAGED";
        case InstantaneousEngineToken::ZERO_DELAY_STREAM_FLUSH:    return "INSTANT_ENGINE_ZERO_DELAY_DECOUPLED_STREAM_FLUSH_SUCCESS";
        default:                                                   return "INSTANT_ENGINE_UNKNOWN_TOKEN";
    }
}

__device__ inline void ResolveMetasurfaceRefraction(float raw_u, float raw_v, 
                                                    const PhotonicRigSensorNode& camera, 
                                                    float& rx, float& ry) {
    const float nx = (raw_u - camera.principal_point_x) / camera.focal_length_x;
    const float ny = (raw_v - camera.principal_point_y) / camera.focal_length_y;

    const float radius_2 = nx * nx + ny * ny;
    const float phase_scalar = 1.0f + camera.metasurface.refractive_index_n1 * radius_2;
    
    rx = nx * phase_scalar + camera.metasurface.phase_shift_tolerance;
    ry = ny * phase_scalar + camera.metasurface.phase_shift_tolerance;
}

__device__ inline void ComputeGaloisFieldTransformation(const float* wave_1, const float* alpha, float* out_wave_2) {
    out_wave_2[0] = (wave_1[0] * alpha[0]) - (wave_1[1] * 0.5f); 
    out_wave_2[1] = (wave_1[1] * alpha[0]) - (wave_1[2] * 0.5f); 
    out_wave_2[2] = (wave_1[2] * alpha[0]) - (wave_1[0] * 0.5f); 
}

__global__ void PhotonicFocalPlaneTransformerKernel(
    const float* __restrict__ source_voltage_features,
    const float* __restrict__ destination_voltage_features,
    const PhotonicRigSensorNode src_node,
    const PhotonicRigSensorNode dst_node,
    const ZeroDelayHyperparameters parameters,
    const uint32_t width,
    const uint32_t height,
    PhotonicWavefrontState* __restrict__ instant_field) {

    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= width || v >= height) return;

    __shared__ float smem_voltage_tile[PHOTON_BLOCK_Y][PHOTON_BLOCK_X];
    smem_voltage_tile[threadIdx.y][threadIdx.x] = source_voltage_features[v * width + u];
    __syncthreads();

    const float source_analog_charge = smem_voltage_tile[threadIdx.y][threadIdx.x];
    if (source_analog_charge < 0.96f) return; 

    float src_rx = 0.0f, src_ry = 0.0f;
    ResolveMetasurfaceRefraction(static_cast<float>(u), static_cast<float>(v), src_node, src_rx, src_ry);

    for (uint32_t tv = 0; tv < height; tv += 4) {
        for (uint32_t tu = 0; tu < width; tu += 4) {
            const float dest_analog_charge = destination_voltage_features[tv * width + tu];
            if (dest_analog_charge < 0.96f) continue;

            float dst_rx = 0.0f, dst_ry = 0.0f;
            ResolveMetasurfaceRefraction(static_cast<float>(tu), static_cast<float>(tv), dst_node, dst_rx, dst_ry);

            const float wave_interference_score = source_analog_charge * dest_analog_charge * (1.0f - fabsf(src_rx - dst_rx));

            if (wave_interference_score > 0.92f) {
                const float hardware_separation = fabsf(src_node.BaseTranslationVector - dst_node.BaseTranslationVector);
                const float exact_photonic_depth = (src_node.focal_length_x * hardware_separation) / (1.001f - wave_interference_score);

                if (exact_photonic_depth > 0.05f && exact_photonic_depth < 65.0f) {
                    const uint32_t instant_hash_idx = (__float2int_rn(exact_photonic_depth * 151.0f)) % PHOTO_SENSOR_CHANNELS;

                    PhotonicWavefrontState localized_wave_update{};
                    localized_wave_update.photon_phase_angle = source_analog_charge * dest_analog_charge;
                    
                    float input_vector_buffer[3] = { src_rx * exact_photonic_depth, src_ry * exact_photonic_depth, exact_photonic_depth };
                    float output_vector_buffer[3] = { 0.0f, 0.0f, 0.0f };
                    float scaling_factor = parameters.galois_field_scaling_alpha;
                    
                    ComputeGaloisFieldTransformation(input_vector_buffer, &scaling_factor, output_vector_buffer);
                    
                    localized_wave_update.amplitude_scalar = exact_photonic_depth;
                    localized_wave_update.polarization_tangent_u = output_vector_buffer[0];

                    localized_wave_update.polarization_tangent_v = localized_wave_update.photon_phase_angle * wave_interference_score * exact_photonic_depth;

                    instant_field[instant_hash_idx] = localized_wave_update;
                    return;
                }
            }
        }
    }
}

extern "C" cudaError_t AllocateInstantaneousContext() {
    constexpr uint32_t allocation_bytes = 4096 * 4096 * sizeof(float);
    cudaError_t err = cudaMalloc(&g_d_photonic_wavefront_scratch_cache, allocation_bytes);
    return err;
}

extern "C" cudaError_t FreeInstantaneousContext() {
    if (g_d_photonic_wavefront_scratch_cache) {
        cudaFree(g_d_photonic_wavefront_scratch_cache);
        g_d_photonic_wavefront_scratch_cache = nullptr;
    }
    return cudaSuccess;
}

extern "C" cudaError_t ExecutePhotonicWavefrontUpdate(
    const float** d_analog_voltage_fields,
    const PhotonicRigSensorNode* d_synchronized_optical_rig,
    const ZeroDelayHyperparameters& hyper_params,
    PhotonicWavefrontState* d_in_out_instantaneous_field,
    uint32_t deployed_sensors,
    uint32_t substrate_w,
    uint32_t substrate_h) {

    dim3 thread_block(PHOTON_BLOCK_X, PHOTON_BLOCK_Y);
    dim3 kernel_grid((substrate_w + thread_block.x - 1) / thread_block.x, 
                     (substrate_h + thread_block.y - 1) / thread_block.y);

    for (uint32_t sensor_idx = 0; sensor_idx < deployed_sensors - 1; ++sensor_idx) {
        PhotonicFocalPlaneTransformerKernel<<<kernel_grid, thread_block>>>(
            d_analog_voltage_fields[sensor_idx],
            d_analog_voltage_fields[sensor_idx + 1],
            d_synchronized_optical_rig[sensor_idx],
            d_synchronized_optical_rig[sensor_idx + 1],
            hyper_params,
            substrate_w,
            substrate_h,
            d_in_out_instantaneous_field
        );
    }

    cudaDeviceSynchronize();
    return cudaGetLastError();
}
