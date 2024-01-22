# establishes policy settings and ensures that the following CMake
# functions are run with a compatible version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(ZAZAM VERSION 1.0)

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
add_subdirectory(/home/df/Documenti/Universita/AMSC/FFT-Carra-Ferrario-Gentile/ffft FFFT)

file(GLOB_RECURSE SOURCES "src/*.cpp" "utils/*.cpp")
add_library(ZAZAM SHARED ${SOURCES})

# Define executable target
add_executable(identify.out examples/identify.cpp)
add_executable(sequence.out examples/sequence.cpp)
add_executable(record.out examples/record.cpp)

find_package(portaudio)
# link the library to link to the executable
target_link_libraries(ZAZAM PUBLIC FFFT) 
target_link_libraries(ZAZAM PUBLIC portaudio) 
target_link_libraries(identify.out PUBLIC ZAZAM) 
target_link_libraries(sequence.out PUBLIC ZAZAM) 
target_link_libraries(record.out PUBLIC ZAZAM) 

# specify the library's header file location
target_include_directories(ZAZAM PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          )

target_include_directories(ZAZAM PUBLIC
                          "${PROJECT_SOURCE_DIR}/AudioFile"
                          )

