#include <iostream>
#include <memory>
#include "ffft.hpp"
#include "../test_template.hpp"
using namespace fftcore;

int main(int argc, char *argv[]){
{
        int n = 1 << 20;
        test_fft<1, SequentialFFT<double>, fftwFFT<double>>(argv);
        /*
        FFTSolver<1> sequential_solver(std::make_unique<SequentialFFT<>>());
        FFTSolver<1> fftw_solver(std::make_unique<fftwFFT<>>());

        CTensorBase<1> tensor_sequential(n);
        tensor_sequential.get_tensor().setRandom();
        
        CTensorBase<1> tensor_fftw_in(tensor_sequential);
        CTensorBase<1> tensor_fftw_out(n);

        sequential_solver.compute_fft(tensor_sequential, FFT_FORWARD); //in-place
        sequential_solver.compute_fft(tensor_sequential, FFT_INVERSE); //in-place

        fftw_solver.compute_fft(tensor_fftw_in, tensor_fftw_out, FFT_FORWARD); //out-of-place
        fftw_solver.compute_fft(tensor_fftw_out, FFT_INVERSE); //in-place

        sequential_solver.get_timer().print();
        fftw_solver.get_timer().print();

        std::cout << "difference between sequential and fftw: " << (tensor_sequential.get_tensor().abs() - tensor_fftw_out.get_tensor().abs()).sum() << std::endl;
        */
    } // 1D-C2C

    {
        int n = 1 << 20;
        FFTSolver<1> fftw_solver(std::make_unique<fftwFFT<>>());

        RTensorBase<1> tensor_fftw_in(n);
        CTensorBase<1> tensor_fftw_out(n);
        tensor_fftw_in.get_tensor().setValues({1,-2,3.5,1.1, -0.9});

        fftw_solver.compute_fft(tensor_fftw_in, tensor_fftw_out, FFT_FORWARD);

        fftw_solver.get_timer().print();

    } // 1D-R2C

    {
    
        int n = 1 << 11;
        FFTSolver<2> fftw_solver(std::make_unique<fftwFFT<>>());

        CTensorBase<2> tensor_fftw_in(n,n);
        CTensorBase<2> tensor_fftw_out(n,n);
        tensor_fftw_in.get_tensor().setRandom();

        fftw_solver.compute_fft(tensor_fftw_in, tensor_fftw_out, FFT_FORWARD); //out-of-place
        fftw_solver.compute_fft(tensor_fftw_out, FFT_INVERSE); //in-place

        fftw_solver.get_timer().print();

        std::cout << "Error (2D-FFTW): " << (tensor_fftw_in.get_tensor().abs() - tensor_fftw_out.get_tensor().abs()).sum() << std::endl;

    } // 2D-C2C

    {
        int n = 1 << 8;

        FFTSolver<3> fftw_solver(std::make_unique<fftwFFT<>>());

        CTensorBase<3> tensor_fftw_in(n,n,n);
        CTensorBase<3> tensor_fftw_out(n,n,n);
        tensor_fftw_in.get_tensor().setRandom();

        fftw_solver.compute_fft(tensor_fftw_in, tensor_fftw_out, FFT_FORWARD); //out-of-place
        fftw_solver.compute_fft(tensor_fftw_out, FFT_INVERSE); //in-place

        fftw_solver.get_timer().print();

        std::cout << "Error (3D-FFTW): " << (tensor_fftw_in.get_tensor().abs() - tensor_fftw_out.get_tensor().abs()).sum() << std::endl;
    } // 3D-C2C
}
