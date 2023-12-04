#include "FFTSolver.hpp"
#include "SequentialFFT.hpp"
#include <iostream>
#include <memory>

using namespace fftcore;


int main(){
    {
        
        FFTSolver<double,1> fft_solver(std::make_unique<SequentialFFT<double>>());
        // TensorFFTBase<std::complex<double>,1> tensor_out_complex(10);
        // TensorFFTBase<std::complex<double>,1> tensor_in_complex(10);
        // TensorFFTBase<double,1> tensor_in_real(10);
        
        // fft_solver.compute_fft(tensor_in_complex, tensor_out_complex, FFT_FORWARD);
        // fft_solver.compute_fft(tensor_in_real, tensor_out_complex, FFT_FORWARD);
        
        TensorFFTBase<std::complex<double>,1> tensor_in_out_complex(4);
        tensor_in_out_complex.get_tensor().setValues({
            {1.0, 0.0},
            {1.0, 0.0},
            {1.0, 0.0},
            {1.0, 0.0},
        });
        //fft_solver.compute_fft(tensor_in_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_out_complex, FFT_INVERSE);
        cout << tensor_in_out_complex.get_tensor() << endl;

    } // 1D 

    {
        FFTSolver<double,2> fft_solver(std::make_unique<SequentialFFT<double>>());
        TensorFFTBase<std::complex<double>,2> tensor_out_complex(10,10);
        TensorFFTBase<std::complex<double>,2> tensor_in_complex(10,10);
        TensorFFTBase<double,2> tensor_in_real(10,10);
        
        fft_solver.compute_fft(tensor_in_complex, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_real, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_complex, FFT_FORWARD);

    }// 2D
    {
        FFTSolver<double,3> fft_solver(std::make_unique<SequentialFFT<double>>());
        TensorFFTBase<std::complex<double>,3> tensor_out_complex(10,10,10);
        TensorFFTBase<std::complex<double>,3> tensor_in_complex(10,10,10);
        TensorFFTBase<double,3> tensor_in_real(10,10,10);
        
        fft_solver.compute_fft(tensor_in_complex, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_real, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_complex, FFT_FORWARD);

    }// 3D


    
}