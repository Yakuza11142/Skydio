import ctypes
from spatial_ai_binding import SpatialAIEngine, CameraExtrinsics, LensDistortionCoefficients, VoxelGridMetadata

def run_test():
    print("[INIT] Loading Spatial AI Library...")
    try:
        engine = SpatialAIEngine("./build/libspatial_ai_perception.so")
    except OSError as e:
        print(f"[ERROR] Could not load compiled binary. Did you build the C++ stack? Details: {e}")
        return

    distortion = LensDistortionCoefficients(k1=-0.2, k2=0.05, p1=0.001, p2=0.002)
    mock_rotation = (ctypes.c_float * 9)(1,0,0, 0,1,0, 0,0,1)
    mock_translation = (ctypes.c_float * 3)(0.0, 0.1, -0.05)
    
    cam_1 = CameraExtrinsics(
        rotation=mock_rotation,
        translation=mock_translation,
        fx=640.0, fy=640.0, cx=320.0, cy=240.0,
        distortion=distortion
    )
    
    metadata = VoxelGridMetadata(
        resolution_meters=0.1,
        origin_x=-20.0, origin_y=-20.0, origin_z=-5.0,
        l_occupancy=0.85, l_free=-0.40, l_min=-2.0, l_max=3.5
    )

    mock_mask_address = ctypes.c_void_p(0) 
    mock_voxel_cache_address = ctypes.c_void_p(0)

    print("[EXECUTE] Triggering Temporal Sparse Projection Pipeline...")
    status = engine.process_frames(
        mask_gpu_pointers=[mock_mask_address],
        extrinsics_list=[cam_1],
        metadata=metadata,
        sparse_voxels_gpu_ptr=mock_voxel_cache_address,
        width=640,
        height=480
    )
    print(f"[SUCCESS] Core entry execution completed with hardware code: {status}")

if __name__ == "__main__":
    run_test()
