#include <iostream>
#include <memory>
#include "ffft.hpp"
#include "../test_template.hpp"
using namespace fftcore;

int main(int argc, char** argv)
{
    test_fft<2, OmpFFT_2D<double>, fftwFFT<double>>(argc, argv); 

    /*
    int x = atoi(argv[1]);
    int n1 = 1 << x;
    x = atoi(argv[2]);
    int n2 = 1 << x;
    constexpr int dim = 2;

    std::cout<<n1<<"x"<<n2<<std::endl;

    FFTSolver<dim> sequential_solver(std::make_unique<SequentialFFT_2D<>>());
    FFTSolver<dim> omp_solver(std::make_unique<OmpFFT_2D<>>());

    CTensorBase<dim> tensor_sequential(n1, n2);
    //tensor_sequential.get_tensor().setRandom();

    for (int i = 0; i < n1; ++i) {
     for (int j = 0; j< n2; ++j) {
           tensor_sequential.get_tensor()(i, j) = std::complex<double>(1,0);//(i, j);
     }
    }

    CTensorBase<dim> tensor_omp(tensor_sequential);

    sequential_solver.compute_fft(tensor_sequential, FFT_FORWARD); //in-place
    sequential_solver.compute_fft(tensor_sequential, FFT_INVERSE); //in-place

    omp_solver.compute_fft(tensor_omp, FFT_FORWARD);
    omp_solver.compute_fft(tensor_omp, FFT_INVERSE);

    sequential_solver.get_timer().print("seq");
    omp_solver.get_timer().print("omp");

    std::cout << "difference : " << (tensor_sequential.get_tensor().abs() - tensor_omp.get_tensor().abs()).sum() << std::endl;
    */
}
