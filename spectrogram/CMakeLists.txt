# Include the ffft library
link_directories(${CMAKE_SOURCE_DIR}/ffft)

# Define executable target for spectrogram
add_executable(spectrogram_app.out spectrogram_main.cpp)
target_link_libraries(spectrogram_app.out PUBLIC FFFT)
target_include_directories(spectrogram_app.out PUBLIC "${CMAKE_SOURCE_DIR}/ffft/include")
