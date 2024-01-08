#include <iostream>
#include <memory>
#include "ffft.hpp"

using namespace fftcore;

int main(int arcg, char** argv)
{

    int x = atoi(argv[1]);
    int n1 = 1 << x;
    x = atoi(argv[2]);
    int n2 = 1 << x;
    constexpr int dim = 2;

    std::cout<<n1<<"x"<<n2<<std::endl;

    FFTSolver<dim> sequential_solver(std::make_unique<SequentialFFT_2D<>>());
    //FFTSolver<dim> fftw_solver(std::make_unique<fftwFFT<>>());

    CTensorBase<dim> tensor_sequential(n1, n2);
    //tensor_sequential.get_tensor().setRandom();
    for (int i = 0; i < n1; ++i) {
     for (int j = 0; j< n2; ++j) {
           tensor_sequential.get_tensor()(i, j) = std::complex<double>(i, j);
     }
    }

    CTensorBase<dim> initial_tensor(tensor_sequential);
    //CTensorBase<dim> tensor_fftw_in(tensor_sequential);
    //CTensorBase<dim> tensor_fftw_out(n1, n2);

    //std::cout<<tensor_sequential.get_tensor()<<std::endl<<std::endl;

    sequential_solver.compute_fft(tensor_sequential, FFT_FORWARD); //in-place

    //std::cout<<tensor_sequential.get_tensor()<<std::endl<<std::endl;
    sequential_solver.compute_fft(tensor_sequential, FFT_INVERSE); //in-place

    //std::cout<<tensor_sequential.get_tensor()<<std::endl<<std::endl;

    //fftw_solver.compute_fft(tensor_fftw_in, tensor_fftw_out, FFT_FORWARD); //out-of-place
    //fftw_solver.compute_fft(tensor_fftw_out, FFT_INVERSE); //in-place

    sequential_solver.get_timer().print();
    //fftw_solver.get_timer().print();

    //std::cout << "difference between sequential and fftw: " << (tensor_sequential.get_tensor().abs() - tensor_fftw_out.get_tensor().abs()).sum() << std::endl;
    std::cout << "difference : " << (tensor_sequential.get_tensor().abs() - initial_tensor.get_tensor().abs()).sum() << std::endl;

}
