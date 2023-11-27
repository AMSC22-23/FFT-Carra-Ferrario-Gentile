//#include "../../fft/src/fftcore/TimeTensor.hpp"
#include "../../fft/src/fftcore/ImageTensor.hpp"
#include "../../fft/src/fftcore/FFTSolver.hpp"
#include "../../fft/src/fftcore/utils/MtxFilesIO.hpp"


#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <iostream>
#include<unsupported/Eigen/SparseExtra>
#include<Eigen/Sparse>
#include<Eigen/Dense>


template <typename T>
void fillTensor(Eigen::Tensor<T, 3>& tensor, const std::vector<T>& values, const std::vector<long>& dimensions) {

    // Set the dimensions of the tensor
    tensor.resize(dimensions);

    // Fill the tensor with values from the vector
    std::copy(values.begin(), values.end(), tensor.data());
}

int main(){
    
    ImageTensor<double> tensor_2d;

    // Load by wrapper
    tensor_2d.load_from_file("./2dTensor.mtx");    
    
    TensorFFTBase<double, 3> tensor_3d;

    // Manual loading
    MtxFilesIO::load_tensor_mtx(tensor_3d.get_tensor(), "./3dTensor.mtx");

    std::cout << tensor_2d.get_tensor() << std::endl;
    std::cout << tensor_3d.get_tensor() << std::endl;


    return 0;
}