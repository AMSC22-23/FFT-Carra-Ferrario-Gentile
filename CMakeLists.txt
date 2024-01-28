# establishes policy settings and ensures that the following CMake
# functions are run with a compatible version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(TEST_FFFT VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define default build type if not provided
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Set build type options
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
  STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

# Set compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add an option for enabling fast math
option(USE_FAST_MATH "Enable fast math optimizations." OFF)

if(USE_FAST_MATH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -funsafe-math-optimizations")
    message(STATUS "Enabled fast math optimizations (-Ofast -funsafe-math-optimizations)")
endif()

# find required packages
find_package(MPI REQUIRED)
cmake_policy(SET CMP0074 NEW)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE HINTS $ENV{mkEigenPrefix})

# Check if CUDA is available and add CUDA source files
find_package(CUDA)
if(CUDA_FOUND)
    #Eigen3 version has to be >= 3.4, otherwise CUDA will not work
    if(Eigen3_VERSION VERSION_LESS 3.4)
        message(WARNING "Eigen3 version is less than 3.4, which is incompatible with CUDA. Disabling CUDA.")
        set(CUDA_FOUND FALSE)
    else()
        # Enable CUDA language
        enable_language(CUDA) #enable NVCC compiler

        # Add CUDA macro for conditional compilation
        add_definitions(-DUSE_CUDA)

        # Set policy to use CUDA_ARCHITECTURES
        cmake_policy(SET CMP0104 NEW)

        # Set CUDA_ARCHITECTURES to compute capability 6.1
        set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75 80 86)
    endif()
endif()

add_subdirectory(ffft)
add_subdirectory(test)
add_subdirectory(spectrogram)

#find portaudio 
find_library(PORTAUDIO_LIBRARY NAMES portaudio)
find_path(PORTAUDIO_INCLUDE_DIR NAMES portaudio.h)

#conditional compilation of zazam
if(PORTAUDIO_LIBRARY AND PORTAUDIO_INCLUDE_DIR)
    message(STATUS "Found portaudio: ${PORTAUDIO_LIBRARY}")
    add_subdirectory(zazam)
else()
    message(STATUS "Could not find portaudio. Disabling zazam.")
endif()