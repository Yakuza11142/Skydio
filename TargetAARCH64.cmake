# Hardware cross-compilation profiling toolkit targeting the Skydio X10 NVIDIA Orin Platform
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Explicit definition mappings for compiler hardware cross pins
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Hardcore optimization configurations flags to maximize hardware instruction sets
set(AARCH64_OPTIMIZATION_FLAGS "-O3 -march=armv8.2-a+crypto+fp16+dotprod -mcpu=cortex-a78ae -mtune=cortex-a78ae -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AARCH64_OPTIMIZATION_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${AARCH64_OPTIMIZATION_FLAGS}" CACHE STRING "" FORCE)

# Direct hardware inclusion paths configuration mapping parameters
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
