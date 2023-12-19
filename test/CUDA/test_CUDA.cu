#define EIGEN_NO_CUDA
#include <iostream>
#include "../../ffft/include/ffft.hpp"
#include "../../ffft/src/fftcore/Strategy/CudaFFT/CudaFFT.cuh"

int main(int arcg, char** argv){
        int x = atoi(argv[1]);
        int n = 1 << x;
        std::cout<<x<<",";
        
        FFTSolver<1> cuda_solver(std::make_unique<CudaFFT<>>());
        FFTSolver<1> fftw_solver(std::make_unique<fftwFFT<>>());
        
        CTensorBase<1> tensor_fftw(n);
        tensor_fftw.get_tensor().setRandom();
        
        CTensorBase<1> tensor_cuda(tensor_fftw);


        cuda_solver.compute_fft(tensor_cuda, FFT_FORWARD);
        double cuda_f = cuda_solver.get_timer().get_last();
        cuda_solver.get_timer().print_last_formatted();
        std::cout<<",";
        cuda_solver.compute_fft(tensor_cuda, FFT_INVERSE);
        double cuda_i = cuda_solver.get_timer().get_last();
        cuda_solver.get_timer().print_last_formatted();
        std::cout<<",";
        
        fftw_solver.compute_fft(tensor_fftw, FFT_FORWARD); 
        double seq_f = fftw_solver.get_timer().get_last();
        fftw_solver.get_timer().print_last_formatted();
        std::cout<<",";
        fftw_solver.compute_fft(tensor_fftw, FFT_INVERSE); 
        double seq_i = fftw_solver.get_timer().get_last();
        fftw_solver.get_timer().print_last_formatted();
        std::cout<<",";
        
        // print speedup
        std::cout<<seq_f/cuda_f<<","<<seq_i/cuda_i<<",";

        std::cout << (tensor_cuda.get_tensor().abs() - tensor_fftw.get_tensor().abs()).sum() << std::endl;
}
