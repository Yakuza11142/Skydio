#ifndef CUDA_SPATIAL_AI_CUH
#define CUDA_SPATIAL_AI_CUH

#include <cuda_runtime.h>
#include <cstdint>

constexpr uint32_t BLOCK_SIZE_BITS = 3; 
constexpr uint32_t BLOCK_VOXEL_COUNT = 512; 
constexpr uint32_t MAX_CAMERAS = 6;
constexpr uint32_t LATENT_FEATURE_DIM = 32;

#ifdef TARGET_HARDWARE_LEGACY_TX2
  constexpr uint32_t MAX_SPARSE_BLOCKS = 1024; 
#else
  constexpr uint32_t MAX_SPARSE_BLOCKS = 4096; 
#endif

struct TensorRTConfig {
    uint32_t input_width;
    uint32_t input_height;
    float detection_threshold;
};

struct LensDistortionCoefficients {
    float k1; 
    float k2;
    float p1; 
    float p2;
};

struct CameraExtrinsics {
    float rotation[9];     // Fixed to 3x3 flattened array matrix
    float translation[3];  // Fixed to 3D offset array vector
    float fx;              
    float fy;              
    float cx;              
    float cy;              
    LensDistortionCoefficients distortion;
};

struct VoxelGridMetadata {
    float resolution_meters;
    float origin_x;
    float origin_y;
    float origin_z;
    float l_occupancy;     
    float l_free;          
    float l_min;           
    float l_max;           
};

extern "C" {
    cudaError_t AllocateSparseAIBuffers();
    cudaError_t FreeSparseAIBuffers();
    
    cudaError_t ExecuteTemporalSparseProjection(
        const float** d_semantic_mask_ptrs,
        const CameraExtrinsics* d_extrinsics,
        const VoxelGridMetadata& grid_meta,
        float* d_in_out_sparse_voxels,
        uint32_t active_cameras,
        uint32_t img_width,
        uint32_t img_height
    );
}

#endif // CUDA_SPATIAL_AI_CUH
