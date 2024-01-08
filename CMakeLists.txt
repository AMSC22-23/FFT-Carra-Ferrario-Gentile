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
add_executable(test_sequential.out test/sequential/test_sequential.cpp)
# link the library to link to the executable
target_link_libraries(test_sequential.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_sequential.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )

# MPI
# Define executable target
add_executable(test_MPI.out test/MPI/test_MPI.cpp)
# link the library to link to the executable
target_link_libraries(test_MPI.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_MPI.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )

# OMP
# Define executable target
add_executable(test_OMP.out test/OMP/test_OMP.cpp)
# link the library to link to the executable
target_link_libraries(test_OMP.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_OMP.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )


#STOCKHAM
# Define executable target
add_executable(test_stockham.out test/stockham/test_stockham.cpp)
# link the library to link to the executable
target_link_libraries(test_stockham.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_stockham.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )

# SEQ_2D
# Define executable target
add_executable(test_sequential_2D.out test/sequential_2D/test_sequential_2D.cpp)
# link the library to link to the executable
target_link_libraries(test_sequential_2D.out PUBLIC FFFT)
# specify the library's header file location
target_include_directories(test_sequential_2D.out PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )