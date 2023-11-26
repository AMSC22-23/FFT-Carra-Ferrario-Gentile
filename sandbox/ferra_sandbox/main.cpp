//#include "../../fft/src/fftcore/TimeTensor.hpp"
#include "../../fft/src/fftcore/ImageTensor.hpp"
#include "../../fft/src/fftcore/FFTSolver.hpp"
//#include "tensorLoading.hpp"

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <iostream>
#include<unsupported/Eigen/SparseExtra>
#include<Eigen/Sparse>
#include<Eigen/Dense>

int main(){
    
    /*
    TimeTensor<int> time_tensor(1);
    std::cout << "Rank: "<<time_tensor.get_rank();
    std::cout << "Empty tensor: "<<time_tensor.get_tensor();
    //TensorFFTBase<double,2> tensorfft2d;
    */

    //EigenTensorFilesIO::load_2d_mtx<double>(tensorfft2d.get_tensor(), "./test.mtx");    
    
    ImageTensor<double> img_tensor;
    img_tensor.load_from_file("./test.mtx");
    std::cout << img_tensor.get_tensor();
    
    fftcore::FFTSolver<double, 3> solver;
    //TensorFFTBase<double, 3> generic_tensor;

    //load_tensor_mtx(generic_tensor.get_tensor(), "./test.mtx");

    return 0;
}