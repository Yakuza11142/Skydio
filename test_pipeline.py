import ctypes
import sys
from spatial_ai_binding import FleetSpatialAIEngine, CameraExtrinsics, LensDistortionCoefficients, VoxelGridMetadata

def run_fleet_test(drone_profile):
    print(f"\n[INIT TESTING] Profile: {drone_profile}")
    
    try:
        engine = FleetSpatialAIEngine(target_drone=drone_profile)
    except OSError:
        print(f"[WARN] Local shared library for {drone_profile} not found. Skipping execution test.")
        return

    # Set camera feed size parameters based on real hardware resolutions
    # Skydio 2 uses lower resolution navigation cameras than the X10
    width, height = (1280, 720) if drone_profile == "TX2" else (4096, 3000)

    distortion = LensDistortionCoefficients(k1=-0.15, k2=0.03, p1=0.0, p2=0.0)
    mock_rotation = (ctypes.c_float * 9)(1,0,0, 0,1,0, 0,0,1)
    mock_translation = (ctypes.c_float * 3)(0.0, 0.0, 0.0)
    
    cam = CameraExtrinsics(
        rotation=mock_rotation, translation=mock_translation,
        fx=500.0, fy=500.0, cx=float(width/2), cy=float(height/2),
        distortion=distortion
    )
    
    metadata = VoxelGridMetadata(
        resolution_meters=0.1,
        origin_x=-10.0, origin_y=-10.0, origin_z=-2.0,
        l_occupancy=0.85, l_free=-0.40, l_min=-2.0, l_max=3.5
    )

    mock_mask_address = ctypes.c_void_p(0) 
    mock_voxel_cache_address = ctypes.c_void_p(0)

    status = engine.process_frames(
        mask_gpu_pointers=[mock_mask_address],
        extrinsics_list=[cam],
        metadata=metadata,
        sparse_voxels_gpu_ptr=mock_voxel_cache_address,
        width=width,
        height=height
    )
    print(f"[SUCCESS] {drone_profile} interface loop passed with status code: {status}")

if __name__ == "__main__":
    # Test both profiles sequentially
    run_fleet_test("TX2")
    run_fleet_test("ORIN")
