# Add the library FFFT
file(GLOB_RECURSE SOURCES "src/*.cpp")

if(CUDA_FOUND)
    # Add CUDA source files to the SOURCES variable
    file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

    list(APPEND SOURCES ${CUDA_SOURCES})
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