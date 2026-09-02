#ifndef CUDA_SPATIAL_AI_SYSTEM_CUH
#define CUDA_SPATIAL_AI_SYSTEM_CUH

#include <cuda_runtime.h>
#include <vector_types.h>
#include <cstdint>

/**
 * @brief Vector-aligned optical lens distortion coefficients.
 * Packed into a unified float4 [k1, k2, p1, p2] to ensure single-cycle register loading.
 */
struct __align__(16) LensDistortionCoefficients {
    float4 k_p_coeffs; 
};

/**
 * @brief Universal Vectorized Camera Calibration & State Block.
 * Strict 16-byte alignment and float4 memory packing guarantee optimal 
 * cache-line performance across NVIDIA Jetson Orin streaming multiprocessors.
 */
struct __align__(16) CameraExtrinsics {
    float4 r0_fx_cx;       // Packed: [r00, r01, r02, fx_scaled_by_inv_width]
    float4 r1_fy_cy;       // Packed: [r10, r11, r12, fy_scaled_by_inv_height]
    float4 r2_trans;       // Packed: [r20, r21, r22, translation_z]
    float4 t_scaled;       // Packed: [translation_x_scaled, translation_y_scaled, cx_scaled, cy_scaled]
    uint32_t sensor_type;  // Sensor profile configuration (0 = Optical, 1 = Thermal)
    uint32_t padding_alignment; // Explicit padding to maintain strict 16-byte alignment
    LensDistortionCoefficients distortion;
};

/**
 * @brief Dynamic Runtime Environmental & Volumetric Stride Voxel Map Configuration.
 */
struct __align__(16) VoxelGridMetadata {
    uint32_t block_dim_bits;    // Volumetric shift tracker (e.g., 3 for 8x8x8 grids)
    uint32_t block_voxel_count; // Total voxels contained inside a unique sparse block
    uint32_t blocks_x;          // Spatial block stride metrics across the X axis
    uint32_t blocks_y;          // Spatial block stride metrics across the Y axis
    uint32_t blocks_z;          // Spatial block stride metrics across the Z axis
    float resolution_meters;    // Voxel geometric dimension limits
    float4 origin_local_shift;  // Packed: [origin_x, origin_y, origin_z, local_offset_x]
    float4 local_shift_y_z;     // Packed: [local_offset_y, local_offset_z, padding, padding]
    float4 bayesian_bounds;     // Packed: [l_occupancy, l_free, l_min, l_max]
};

extern "C" {
    /**
     * @brief Raw C-Linkage Entry Point for the Vectorized Runtime Kernel Launcher.
     */
    cudaError_t ExecutePeakSpatialProjection(
        const cudaTextureObject_t* d_semantic_tex_objs,
        const CameraExtrinsics* d_extrinsics,
        const VoxelGridMetadata* d_grid_meta,
        float* d_in_out_sparse_voxels,
        uint32_t active_cameras,
        const uint32_t* d_active_block_indices,
        uint32_t total_active_blocks,
        cudaStream_t stream
    );
}

#endif // CUDA_SPATIAL_AI_SYSTEM_CUH
