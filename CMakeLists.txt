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

# Include ffft library
add_subdirectory(ffft)

# SEQUENTIAL
# Define executable target
add_executable(test_sequential.out test/1D/sequential/test_sequential.cpp)
# link the library to link to the executable
target_link_libraries(test_sequential.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_sequential.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )

# MPI
# Define executable target
add_executable(test_MPI.out test/1D/MPI/test_MPI.cpp)
# link the library to link to the executable
target_link_libraries(test_MPI.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_MPI.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )

# OMP
# Define executable target
add_executable(test_OMP.out test/1D/OMP/test_OMP.cpp)
# link the library to link to the executable
target_link_libraries(test_OMP.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_OMP.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )


#STOCKHAM
# Define executable target
add_executable(test_stockham.out test/1D/stockham/test_stockham.cpp)
# link the library to link to the executable
target_link_libraries(test_stockham.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_stockham.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )

# SEQ_2D
# Define executable target
add_executable(test_sequential_2D.out test/2D/sequential_2D/test_sequential_2D.cpp)
# link the library to link to the executable
target_link_libraries(test_sequential_2D.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_sequential_2D.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )

# OMP_2D
# Define executable target
add_executable(test_OMP_2D.out test/2D/OMP_2D/test_OMP_2D.cpp)
# link the library to link to the executable
target_link_libraries(test_OMP_2D.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_OMP_2D.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )                    
# MPI_2D
# Define executable target
add_executable(test_MPI_2D.out test/2D/MPI_2D/test_MPI_2D.cpp)
# link the library to link to the executable
target_link_libraries(test_MPI_2D.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_MPI_2D.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )

# MPI_3D
# Define executable target
add_executable(test_MPI_3D.out test/3D/MPI_3D/test_MPI_3D.cpp)
# link the library to link to the executable
target_link_libraries(test_MPI_3D.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_MPI_3D.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )
# CUDA
#find_package(CUDA)
if(CUDA_FOUND)
    enable_language(CUDA) #enable NVCC compiler
    
    #set policy to use CUDA_ARCHITECTURES
    cmake_policy(SET CMP0104 NEW)

    #set CUDA_ARCHITECTURES to compute capability 6.1
    set(CMAKE_CUDA_ARCHITECTURES 61 62 70 72 75 80 86)

    # Define executable target
    add_executable(test_CUDA.out test/1D/CUDA/test_CUDA.cu)

    # link the library to link to the executable
    target_link_libraries(test_CUDA.out PUBLIC FFFT cufft)

    # specify the library's header file location
    target_include_directories(test_CUDA.out PUBLIC
                            "${PROJECT_SOURCE_DIR}/ffft/include"
                            )
endif()

# SEQ_3D
# Define executable target
add_executable(test_sequential_3D.out test/3D/sequential_3D/test_sequential_3D.cpp)
# link the library to link to the executable
target_link_libraries(test_sequential_3D.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_sequential_3D.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )

# OMP_3D
# Define executable target
add_executable(test_OMP_3D.out test/3D/OMP_3D/test_OMP_3D.cpp)
# link the library to link to the executable
target_link_libraries(test_OMP_3D.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_OMP_3D.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )   

# SPECTROGRAM
# Define executable target
add_executable(spectrogram.out spectrogram/spectrogram_main.cpp)
# link the library to link to the executable
target_link_libraries(spectrogram.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(spectrogram.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )

