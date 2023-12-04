#include "../fft/src/fftcore/TensorFFTBase.hpp"
#include "../fft/src/fftcore/SequentialFFT.hpp"
#include "../fft/src/fftcore/MPIFFT.hpp"
#include "../fft/src/fftcore/FFTSolver.hpp"
#include "../fft/src/fftcore/FFTSolver.hpp"

#include <memory>
#include <mpi.h>

using namespace fftcore;

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    FFTSolver<1> fft_solver(std::make_unique<MPIFFT<>>());

    CTensorBase<1> tensor_in_out_complex(4);
    tensor_in_out_complex.get_tensor().setValues({
        {1.0, 1.0},
        {-10.0, 2.0},
        {-10.0, -3.0},
        {-2.0, 4.0},

    });
    //operazioni su w
    fft_solver.compute_fft(tensor_in_out_complex,FFT_FORWARD);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0)
        cout << tensor_in_out_complex.get_tensor();
    MPI_Finalize();

}