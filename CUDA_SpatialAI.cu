#include "CUDA_SpatialAI.cuh"
#include <device_launch_parameters.h>

__device__ uint32_t Morton3D(uint32_t x, uint32_t y, uint32_t z) {
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x <<  8)) & 0x0300F00F;
    x = (x | (x <<  4)) & 0x030C30C3;
    x = (x | (x <<  2)) & 0x09249249;

    y = (y | (y << 16)) & 0x030000FF;
    y = (y | (y <<  8)) & 0x0300F00F;
    y = (y | (y <<  4)) & 0x030C30C3;
    y = (y | (y <<  2)) & 0x09249249;

    z = (z | (z << 16)) & 0x030000FF;
    z = (z | (z <<  8)) & 0x0300F00F;
    z = (z | (z <<  4)) & 0x030C30C3;
    z = (z | (z <<  2)) & 0x09249249;

    return x | (y << 1) | (z << 2);
}

__device__ void ApplyBrownConradyDistortion(float x_norm, float y_norm, 
                                            const LensDistortionCoefficients& dist,
                                            float& x_dist, float& y_dist) {
    float r2 = x_norm * x_norm + y_norm * y_norm;
    float r4 = r2 * r2;
    float radial = 1.0f + dist.k1 * r2 + dist.k2 * r4;
    float dx = 2.0f * dist.p1 * x_norm * y_norm + dist.p2 * (r2 + 2.0f * x_norm * x_norm);
    float dy = dist.p1 * (r2 + 2.0f * y_norm * y_norm) + 2.0f * dist.p2 * x_norm * y_norm;
    x_dist = x_norm * radial + dx;
    y_dist = y_norm * radial + dy;
}

__global__ void SparseTemporalProjectionKernel(
    const float** __restrict__ d_semantic_mask_ptrs,
    const CameraExtrinsics* __restrict__ d_extrinsics,
    VoxelGridMetadata grid_meta,
    float* __restrict__ d_in_out_sparse_voxels,
    uint32_t active_cameras,
    uint32_t img_width,
    uint32_t img_height) 
{
    uint32_t block_idx = blockIdx.x;
    if (block_idx >= MAX_SPARSE_BLOCKS) return;
    
    uint32_t tid = threadIdx.x; 
    if (tid >= BLOCK_VOXEL_COUNT) return;

    uint32_t vx = tid & 0x7;
    uint32_t vy = (tid >> 3) & 0x7;
    uint32_t vz = (tid >> 6) & 0x7;

    float gx = grid_meta.origin_x + (block_idx * 8 + vx) * grid_meta.resolution_meters;
    float gy = grid_meta.origin_y + (block_idx * 8 + vy) * grid_meta.resolution_meters;
    float gz = grid_meta.origin_z + (block_idx * 8 + vz) * grid_meta.resolution_meters;

    for (uint32_t cam = 0; cam < active_cameras; ++cam) {
        CameraExtrinsics ext = d_extrinsics[cam];

        // Fully array-indexed matrix transformations for rigid body rotation
        float cx_space = ext.rotation[0]*gx + ext.rotation[1]*gy + ext.rotation[2]*gz + ext.translation[0];
        float cy_space = ext.rotation[3]*gx + ext.rotation[4]*gy + ext.rotation[5]*gz + ext.translation[1];
        float cz_space = ext.rotation[6]*gx + ext.rotation[7]*gy + ext.rotation[8]*gz + ext.translation[2];

        if (cz_space <= 0.1f) continue;

        float x_norm = cx_space / cz_space;
        float y_norm = cy_space / cz_space;

        float x_dist, y_dist;
        ApplyBrownConradyDistortion(x_norm, y_norm, ext.distortion, x_dist, y_dist);

        int u = static_cast<int>(ext.fx * x_dist + ext.cx);
        int v = static_cast<int>(ext.fy * y_dist + ext.cy);

        if (u >= 0 && u < img_width && v >= 0 && v < img_height) {
            float prob = d_semantic_mask_ptrs[cam][v * img_width + u];
            float avg_prob = prob; // Isolated thread evaluation 
            float update_log_odds = (avg_prob > 0.5f) ? grid_meta.l_occupancy : grid_meta.l_free;

            uint32_t global_voxel_idx = (block_idx * BLOCK_VOXEL_COUNT) + Morton3D(vx, vy, vz);
            
            float current_val = d_in_out_sparse_voxels[global_voxel_idx];
            float next_val = fminf(fmaxf(current_val + update_log_odds, grid_meta.l_min), grid_meta.l_max);
            d_in_out_sparse_voxels[global_voxel_idx] = next_val;
        }
    }
}

extern "C" {
    cudaError_t AllocateSparseAIBuffers() { return cudaSuccess; }
    cudaError_t FreeSparseAIBuffers() { return cudaSuccess; }

    cudaError_t ExecuteTemporalSparseProjection(
        const float** d_semantic_mask_ptrs,
        const CameraExtrinsics* d_extrinsics,
        const VoxelGridMetadata& grid_meta,
        float* d_in_out_sparse_voxels,
        uint32_t active_cameras,
        uint32_t img_width,
        uint32_t img_height) 
    {
        uint32_t blocks = MAX_SPARSE_BLOCKS;
        uint32_t threads = BLOCK_VOXEL_COUNT;

        SparseTemporalProjectionKernel<<<blocks, threads>>>(
            d_semantic_mask_ptrs,
            d_extrinsics,
            grid_meta,
            d_in_out_sparse_voxels,
            active_cameras,
            img_width,
            img_height
        );
        return cudaGetLastError();
    }
}
