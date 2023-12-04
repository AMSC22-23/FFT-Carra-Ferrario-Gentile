#include "FFTSolver.hpp"
#include "SequentialFFT.hpp"
#include <iostream>
#include <memory>

using namespace fftcore;

int main(){
    {
        
        FFTSolver<1> fft_solver(std::make_unique<SequentialFFT<>>());
        // TensorFFTBase<std::complex<double>,1> tensor_out_complex(10);
        // TensorFFTBase<std::complex<double>,1> tensor_in_complex(10);
        // TensorFFTBase<double,1> tensor_in_real(10);
        
        // fft_solver.compute_fft(tensor_in_complex, tensor_out_complex, FFT_FORWARD);
        // fft_solver.compute_fft(tensor_in_real, tensor_out_complex, FFT_FORWARD);

        CTensorBase<1> tensor_in_out_complex(8);
        tensor_in_out_complex.get_tensor().setValues({
            {1.0, 1.0},
            {-10.0, 2.0},
            {-10.0, -3.0},
            {-2.0, 4.0},
            {1.0, 1.0},
            {-10.0, 2.0},
            {-10.0, -3.0},
            {-2.0, 4.0},
        });
       //fft_solver.compute_fft(tensor_in_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_out_complex, FFT_INVERSE);
        cout << tensor_in_out_complex.get_tensor() << endl;

    } // 1D 

    {
        FFTSolver<2> fft_solver(std::make_unique<SequentialFFT<>>());
        CTensorBase<2> tensor_out_complex(10,10);
        CTensorBase<2> tensor_in_complex(10,10);
        RTensorBase<2> tensor_in_real(10,10);
        
        fft_solver.compute_fft(tensor_in_complex, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_real, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_complex, FFT_FORWARD);

    }// 2D
    {
        FFTSolver<3> fft_solver(std::make_unique<SequentialFFT<>>());
        CTensorBase<3> tensor_out_complex(10,10,10);
        CTensorBase<3> tensor_in_complex(10,10,10);
        RTensorBase<3> tensor_in_real(10,10,10);
        
        fft_solver.compute_fft(tensor_in_complex, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_real, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_complex, FFT_FORWARD);

    }// 3D



}