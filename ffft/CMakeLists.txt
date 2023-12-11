# Add the library FFT

file(GLOB_RECURSE SOURCES "src/*.cpp")

# Create FFFT target
add_library(FFFT SHARED ${SOURCES})

#include fftw3
target_include_directories(FFFT PUBLIC "${FFTW3_DIR}/include")
# Link directories for the FFTW3 library
target_link_directories(FFFT PUBLIC "${FFTW3_DIR}/lib")

# include MPI and Eigen
find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE
		HINTS $ENV{mkEigenPrefix})

# Link libraries to the executable
target_link_libraries(FFFT PUBLIC fftw3 MPI::MPI_CXX Eigen3::Eigen)

# Additional settings for MPI
target_include_directories(FFFT PUBLIC ${MPI_C_INCLUDE_PATH})