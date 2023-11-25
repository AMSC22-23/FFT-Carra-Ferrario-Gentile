#include "../../fft/src/fftcore/TimeTensor.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <iostream>

int main(){
    TimeTensor<int, 3> time_tensor(1, 1, 1);
    std::cout << time_tensor.get_rank();
    std::cout << time_tensor.get_tensor();


    return 0;
}