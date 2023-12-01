#include "../edo_sandbox/fftcore/TensorFFTBase.hpp"
#include "../edo_sandbox/fftcore/SequentialFFT.hpp"
#include "../edo_sandbox/fftcore/FFTSolver.hpp"
#include <memory>

using namespace fftcore;

int main(){
    FFTSolver<double,1> fft_solver(std::make_unique<SequentialFFT<double, 1>>());
    TensorFFTBase<double,1> tensor_out(10);
    TensorFFTBase<double,1> tensor_in(10);

    //operazioni su w
    fft_solver.compute_fft_C2C(tensor_in, tensor_out,FFT_FORWARD);
}