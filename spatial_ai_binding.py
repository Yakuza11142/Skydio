import ctypes

class LensDistortionCoefficients(ctypes.Structure):
    _fields_ = [
        ("k1", ctypes.c_float),
        ("k2", ctypes.c_float),
        ("p1", ctypes.c_float),
        ("p2", ctypes.c_float)
    ]

class CameraExtrinsics(ctypes.Structure):
    _fields_ = [
        ("rotation", ctypes.c_float * 9),    
        ("translation", ctypes.c_float * 3), 
        ("fx", ctypes.c_float),
        ("fy", ctypes.c_float),
        ("cx", ctypes.c_float),
        ("cy", ctypes.c_float),
        ("distortion", LensDistortionCoefficients)
    ]

class VoxelGridMetadata(ctypes.Structure):
    _fields_ = [
        ("resolution_meters", ctypes.c_float),
        ("origin_x", ctypes.c_float),
        ("origin_y", ctypes.c_float),
        ("origin_z", ctypes.c_float),
        ("l_occupancy", ctypes.c_float),
        ("l_free", ctypes.c_float),
        ("l_min", ctypes.c_float),
        ("l_max", ctypes.c_float)
    ]

class SpatialAIEngine:
    def __init__(self, shared_object_path="./build/libspatial_ai_perception.so"):
        self.lib = ctypes.CDLL(shared_object_path)
        
        self.lib.AllocateSparseAIBuffers.restype = ctypes.c_int
        self.lib.FreeSparseAIBuffers.restype = ctypes.c_int
        
        self.lib.ExecuteTemporalSparseProjection.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),          
            ctypes.POINTER(CameraExtrinsics),          
            ctypes.POINTER(VoxelGridMetadata),         
            ctypes.POINTER(ctypes.c_float),            
            ctypes.c_uint32,                           
            ctypes.c_uint32,                           
            ctypes.c_uint32                            
        ]
        self.lib.ExecuteTemporalSparseProjection.restype = ctypes.c_int

    def allocate(self):
        return self.lib.AllocateSparseAIBuffers()

    def free(self):
        return self.lib.FreeSparseAIBuffers()

    def process_frames(self, mask_gpu_pointers, extrinsics_list, metadata, sparse_voxels_gpu_ptr, width, height):
        active_cams = len(extrinsics_list)
        
        pointer_array_type = ctypes.c_void_p * active_cams
        d_mask_ptrs = pointer_array_type(*mask_gpu_pointers)
        
        extrinsics_array_type = CameraExtrinsics * active_cams
        d_ext = extrinsics_array_type(*extrinsics_list)
        
        status = self.lib.ExecuteTemporalSparseProjection(
            ctypes.cast(d_mask_ptrs, ctypes.POINTER(ctypes.c_void_p)),
            d_ext,
            ctypes.byref(metadata),
            ctypes.cast(sparse_voxels_gpu_ptr, ctypes.POINTER(ctypes.c_float)),
            active_cams,
            width,
            height
        )
        return status
