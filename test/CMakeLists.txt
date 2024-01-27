# Include the ffft library
link_directories(${CMAKE_SOURCE_DIR}/ffft)

# Macro for adding test executables
macro(add_test_executable TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} PUBLIC FFFT)
    target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/ffft/include)
endmacro()

# Add the test executables
add_test_executable(test_sequential.out "1D/sequential/test_sequential.cpp")
add_test_executable(test_MPI.out "1D/MPI/test_MPI.cpp")
add_test_executable(test_OMP.out "1D/OMP/test_OMP.cpp")
add_test_executable(test_stockham.out "1D/stockham/test_stockham.cpp")

add_test_executable(test_sequential_2D.out "2D/sequential_2D/test_sequential_2D.cpp")
add_test_executable(test_MPI_2D.out "2D/MPI_2D/test_MPI_2D.cpp")
add_test_executable(test_OMP_2D.out "2D/OMP_2D/test_OMP_2D.cpp")
add_test_executable(test_MPI_OMP_2D.out "2D/MPI_OMP_2D/test_MPI_OMP_2D.cpp")

add_test_executable(test_sequential_3D.out "3D/sequential_3D/test_sequential_3D.cpp")
add_test_executable(test_MPI_3D.out "3D/MPI_3D/test_MPI_3D.cpp")
add_test_executable(test_OMP_3D.out "3D/OMP_3D/test_OMP_3D.cpp")

if(CUDA_FOUND)
    add_test_executable(test_CUDA.out "1D/CUDA/test_CUDA.cpp")
endif()