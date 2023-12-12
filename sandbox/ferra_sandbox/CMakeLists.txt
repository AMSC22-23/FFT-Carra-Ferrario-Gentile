cmake_minimum_required(VERSION 3.10)

# Project name and version
project(FFT-CARRA-FERRARIO-GENTILE VERSION 1.0)

# C++ standard
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

# Set the FFTW3 path
#set(FFTW3_DIR "/opt/homebrew/Cellar/fftw/3.3.10_1")

# Include directories for FFTW3 and other required libraries
include_directories(
    "${PROJECT_SOURCE_DIR}/src"
    "${FFTW3_DIR}/include"
    # Add other required include directories here
)

# Link directories for the FFTW3 library
#link_directories("${FFTW3_DIR}/lib")

# Find required packages
find_package(MPI REQUIRED)
#find_package(Eigen3 REQUIRED NO_MODULE)

# Source files
#file(GLOB_RECURSE SOURCES "../../fft/src/*.cpp")
include_directories(main "../../eigen")

# Define executable target
add_executable(main.out main.cpp ../../ffft/src/fftcore/Timer/Timer.cpp)

# Link libraries to the executable
target_link_libraries(main.out PRIVATE MPI::MPI_CXX)

# Set the location for the object files
set_target_properties(main.out PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/obj"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/obj"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
)

# Additional settings for MPI
target_include_directories(main.out PRIVATE ${MPI_C_INCLUDE_PATH})