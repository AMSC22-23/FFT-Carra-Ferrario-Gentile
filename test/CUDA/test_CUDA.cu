#define EIGEN_NO_CUDA
#include <iostream>
#include <memory>
#include "../../ffft/include/ffft.hpp"

#include "../../ffft/src/fftcore/Strategy/CudaFFT/CudaFFT.cuh" //temporary

using namespace fftcore;

int main(void){

    //set random seed
    srand(time(NULL));

    int n = 1 << 25;
    CTensorBase<1> tensor_fftw(n);
    tensor_fftw.get_tensor().setRandom();

    CTensorBase<1> tensor_cuda(tensor_fftw.get_tensor());
    FFTSolver<1> fftw_solver(std::make_unique<fftwFFT<>>());

    fftw_solver.compute_fft(tensor_fftw, FFT_FORWARD);

    fftw_solver.get_timer().print("fftw");

    FFTSolver<1> cuda_solver(std::make_unique<CudaFFT<>>());

    cudaFree(0);
    cuda_solver.compute_fft(tensor_cuda, FFT_FORWARD);

    cuda_solver.get_timer().print("Cuda");

    std::cout << "Error: " << (tensor_fftw.get_tensor() - tensor_cuda.get_tensor()).abs().sum() << std::endl;

    //print first 10 elements of both vectors
    std::cout << "fftw: " << std::endl;
    for(int i = 0; i < 10; i++){
        std::cout << tensor_fftw.get_tensor()(i) << std::endl;
    }

    std::cout << "Cuda: " << std::endl;
    for(int i = 0; i < 10; i++){
        std::cout << tensor_cuda.get_tensor()(i) << std::endl;
    }

    //print difference between first 10 elements of both vectors
    std::cout << "Difference: " << std::endl;
    for(int i = 0; i < 10; i++){
        std::cout << tensor_fftw.get_tensor()(i) - tensor_cuda.get_tensor()(i) << std::endl;
    }


}