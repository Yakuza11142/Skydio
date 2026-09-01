cmake_minimum_required(VERSION 3.22)
# FIX: Removed the invalid "PatchSuite" keyword from the language declaration
project(SkydioX10ProEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(TargetAARCH64.cmake)
include(TensorRTOptimizer.cmake)

# Direct definitions setup to establish multi-fleet single codebase hardware profiles
add_compile_definitions(TARGET_HARDWARE_ORIN)

# Include directories layout
include_directories(
    src/mod_1_spatial_vision/include
    src/mod_2_thermal_safety/include
    src/mod_3_hardened_clock/include
    src/mod_4_ota_verification/include
)

# Shared Memory Engine Core target definition
add_library(skydio_pro_core SHARED
    src/mod_1_spatial_vision/src/CUDA_SpatialAI.cu
    src/mod_1_spatial_vision/src/TensorRTEngine.cpp
    src/mod_1_spatial_vision/src/InstantaneousFieldOrchestrator.cpp
    src/mod_2_thermal_safety/src/QuantumThermalHAL.cpp
    src/mod_2_thermal_safety/src/LiouvilleThermalField.cpp
    src/mod_2_thermal_safety/src/MotorSafetyDaemon.cpp
    src/mod_3_hardened_clock/src/MonotonicClockHAL.cpp
    src/mod_3_hardened_clock/src/EntangledTelemetry.cpp
    src/mod_3_hardened_clock/src/PacketRingBuffer.cpp
    src/mod_3_hardened_clock/src/TelemetryGateway.cpp
    src/mod_4_ota_verification/src/DriverValidator.cpp
    src/mod_4_ota_verification/src/RealTimeWatchdog.cpp
    src/mod_4_ota_verification/src/ota_entry.cpp
)

optimize_neural_target(skydio_pro_core)

# Automated Verification Test Binaries execution setup
add_executable(test_vision test_cuda_wire_detection.cpp)
target_link_libraries(test_vision PRIVATE skydio_pro_core)

add_executable(test_thermal test_thermal_clamping.cpp)
target_link_libraries(test_thermal PRIVATE skydio_pro_core)

add_executable(test_clock test_clock_overflow_immunity.cpp)
target_link_libraries(test_clock PRIVATE skydio_pro_core)
