#include <iostream>
#include "ffft.hpp"

int main(){
        int n = 1 << 25;

        FFTSolver<1> omp_solver(std::make_unique<OmpFFT<>>());
        FFTSolver<1> sequential_solver(std::make_unique<SequentialFFT<>>());
        FFTSolver<1> fftw_solver(std::make_unique<fftwFFT<>>());

        CTensorBase<1> tensor_sequential(n);
        tensor_sequential.get_tensor().setRandom();
        
        CTensorBase<1> tensor_omp(tensor_sequential);

        CTensorBase<1> tensor_fftw_in(tensor_sequential);
        CTensorBase<1> tensor_fftw_out(n);

        omp_solver.compute_fft(tensor_omp, FFT_FORWARD);
        omp_solver.compute_fft(tensor_omp, FFT_INVERSE);
        
        sequential_solver.compute_fft(tensor_sequential, FFT_FORWARD); 
        sequential_solver.compute_fft(tensor_sequential, FFT_INVERSE); 

        fftw_solver.compute_fft(tensor_fftw_in, tensor_fftw_out, FFT_FORWARD); //out-of-place
        fftw_solver.compute_fft(tensor_fftw_out, FFT_INVERSE); //in-place

        omp_solver.get_timer().print("OMP");
        sequential_solver.get_timer().print("Sequential");
        fftw_solver.get_timer().print("FFTW");

        std::cout << "difference between omp and fftw: " << (tensor_omp.get_tensor().abs() - tensor_sequential.get_tensor().abs()).sum() << std::endl;
}
