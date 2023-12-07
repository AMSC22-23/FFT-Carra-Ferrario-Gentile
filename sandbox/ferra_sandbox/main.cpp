#include "../../fft/src/fftcore/Tensor/TensorFFTBase.hpp"
#include "../../fft/src/fftcore/Strategy/SequentialFFT/SequentialFFT.hpp"
#include "../../fft/src/fftcore/Strategy/MPIFFT/MPIFFT.hpp"
#include "../../fft/src/fftcore/FFTSolver.hpp"

#include <memory>

using namespace fftcore;

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    FFTSolver<1, float> fft_solver(std::make_unique<MPIFFT<float>>());

    CTensorBase<1, float> tensorA(4);
    
    tensorA.get_tensor().setValues({
        {1.0, 1.0},
        {2.0, 1.0},
        {4.0, 1.0},
        {3.0, 1.0},
    });

    //operazioni su w
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    

    fft_solver.compute_fft(tensorA,FFT_FORWARD);    
    cout << "Final: " << tensorA.get_tensor() << endl;

    fft_solver.compute_fft(tensorA,FFT_INVERSE);

    cout << "Final: " << tensorA.get_tensor() << endl;

    MPI_Finalize();

}