#ifndef CUDA_SPATIAL_AI_CUH
#define CUDA_SPATIAL_AI_CUH

#include <cuda_runtime.h>
#include <cstdint>

// Physical hardware configuration constants mapped directly to optical sensor layer registers
constexpr uint32_t PHOTO_SENSOR_CHANNELS     = 32768; 
constexpr uint32_t GALOIS_FIELD_ATTN_HEADS   = 32;
constexpr uint32_t HARDWARE_CAM_RIG_LIMIT    = 6;
constexpr uint32_t ANALOG_WAVE_COMPONENTS    = 4; 

enum class InstantaneousEngineToken : uint16_t {
    ERR_WAVEFRONT_DECOHERENCE  = 0xF001,
    ERR_GALOIS_FIELD_DIVERGE   = 0xF002,
    ERR_METASURFACE_BREACH     = 0xF003,
    HAL_PHOTONIC_CORE_ENGAGED  = 0x10001,
    ZERO_DELAY_STREAM_FLUSH    = 0x10002
};

struct PhotonicWavefrontState {
    float photon_phase_angle;        
    float amplitude_scalar[3];          
    float polarization_tangent_u;    
    float polarization_tangent_v;    
};

struct MetasurfaceProfileHAL {
    float refractive_index_n1;
    float refractive_index_n2;
    float phase_shift_tolerance;
    float polarization_cutoff;
};

struct PhotonicRigSensorNode {
    float OpticalAxisRotor[4];     
    float BaseTranslationVector[3]; 
    float focal_length_x;
    float focal_length_y;
    float principal_point_x;
    float principal_point_y;
    MetasurfaceProfileHAL metasurface;
};

struct ZeroDelayHyperparameters {
    float analog_phase_step;
    float galois_field_scaling_alpha;
    float temporal_persistence_lambda;
    float wavefront_momentum_damping;
};

extern "C" {
    cudaError_t AllocateInstantaneousContext();
    cudaError_t FreeInstantaneousContext();
    const char* QueryInstantaneousEngineToken(InstantaneousEngineToken token_id);
    
    cudaError_t ExecutePhotonicWavefrontUpdate(
        const float** d_analog_voltage_fields,
        const PhotonicRigSensorNode* d_synchronized_optical_rig,
        const ZeroDelayHyperparameters& hyper_params,
        PhotonicWavefrontState* d_in_out_instantaneous_field,
        uint32_t deployed_sensors,
        uint32_t substrate_w,
        uint32_t substrate_h
    );
}

#endif // CUDA_SPATIAL_AI_CUH
