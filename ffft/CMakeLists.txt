# Add the library FFT

file(GLOB_RECURSE SOURCES "src/*.cpp")

# Check if CUDA is available and add CUDA source files
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA) #enable NVCC compiler

    # Set policy to use CUDA_ARCHITECTURES
    cmake_policy(SET CMP0104 NEW)

    # Set CUDA_ARCHITECTURES to compute capability 6.1
    set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75 80 86)

    # Add CUDA source files to the SOURCES variable
    file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

    # Remove a specific cu file from the list
    list(REMOVE_ITEM CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/fftcore/Strategy/1D/cufftFFT/cufftFFT.cu")

    list(APPEND SOURCES ${CUDA_SOURCES})

endif()

# Create FFFT target
add_library(FFFT SHARED ${SOURCES})

#include fftw3
target_include_directories(FFFT PUBLIC "${FFTW3_DIR}/include")
# Link directories for the FFTW3 library
target_link_directories(FFFT PUBLIC "${FFTW3_DIR}/lib")

# include OpenMP, MPI and Eigen
find_package(MPI REQUIRED)
cmake_policy(SET CMP0074 NEW)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE
		HINTS $ENV{mkEigenPrefix})


# Link libraries to the executable
target_link_libraries(FFFT PUBLIC fftw3 MPI::MPI_CXX Eigen3::Eigen OpenMP::OpenMP_CXX)

# Additional settings for MPI
target_include_directories(FFFT PUBLIC ${MPI_C_INCLUDE_PATH})