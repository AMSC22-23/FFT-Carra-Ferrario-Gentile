# Add the library FFT

file(GLOB_RECURSE SOURCES "src/*.cpp")

# include OpenMP, MPI and Eigen
find_package(MPI REQUIRED)
cmake_policy(SET CMP0074 NEW)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE
		HINTS $ENV{mkEigenPrefix})

# Check if CUDA is available and add CUDA source files
find_package(CUDA QUIET)
if(CUDA_FOUND)

    #Eigen3 version has to be >= 3.4, otherwise CUDA will not work
    if(Eigen3_VERSION VERSION_LESS 3.4)
        message(WARNING "Eigen3 version is less than 3.4, which is incompatible with CUDA. Disabling CUDA.")
        set(CUDA_FOUND FALSE)
    else()
        # Enable CUDA language
        enable_language(CUDA) #enable NVCC compiler

        # Set policy to use CUDA_ARCHITECTURES
        cmake_policy(SET CMP0104 NEW)

        # Set CUDA_ARCHITECTURES to compute capability 6.1
        set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75 80 86)

        # Add CUDA source files to the SOURCES variable
        file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

        list(APPEND SOURCES ${CUDA_SOURCES})
    endif()
endif()

# Create FFFT target
add_library(FFFT SHARED ${SOURCES})

#include fftw3
target_include_directories(FFFT PUBLIC "${FFTW3_DIR}/include")
# Link directories for the FFTW3 library
target_link_directories(FFFT PUBLIC "${FFTW3_DIR}/lib")


# Link libraries to the executable
target_link_libraries(FFFT PUBLIC fftw3 MPI::MPI_CXX Eigen3::Eigen OpenMP::OpenMP_CXX)

# Additional settings for MPI
target_include_directories(FFFT PUBLIC ${MPI_C_INCLUDE_PATH})