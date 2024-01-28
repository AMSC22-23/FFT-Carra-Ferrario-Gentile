# Collect all the source files for the ZAZAM library
file(GLOB_RECURSE ZAZAM_SOURCES "src/*.cpp" "utils/*.cpp")

# Create a static library with the common source files
add_library(zazam_lib STATIC ${ZAZAM_SOURCES})

# Link the static library with the FFFT and portaudio libraries
target_link_libraries(zazam_lib PUBLIC FFFT ${PORTAUDIO_LIBRARY})

# Include the portaudio headers for the library
target_include_directories(zazam_lib PUBLIC ${PORTAUDIO_INCLUDE_DIR})

# Specify the include directories for the library
target_include_directories(zazam_lib PUBLIC
                          "${PROJECT_SOURCE_DIR}/ffft/include"
                          "${PROJECT_SOURCE_DIR}/AudioFile"
                          )

# Define executable targets
add_executable(identify.out examples/identify.cpp)
add_executable(sequence.out examples/sequence.cpp)
add_executable(record.out examples/record.cpp)

# Link the executables with the static library
target_link_libraries(identify.out PUBLIC zazam_lib)
target_link_libraries(sequence.out PUBLIC zazam_lib)
target_link_libraries(record.out PUBLIC zazam_lib)