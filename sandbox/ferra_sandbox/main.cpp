#include "../../fft/src/fftcore/Tensor/TensorFFTBase.hpp"
#include "../../fft/src/fftcore/Strategy/SequentialFFT/SequentialFFT.hpp"
#include "../../fft/src/fftcore/Strategy/MPIFFT/MPIFFT.hpp"
#include "../../fft/src/fftcore/FFTSolver.hpp"

#include <memory>
#include <cmath>

using namespace fftcore;


bool tensor_equality(CTensorBase<1> a, CTensorBase<1> b){
    int n = a.get_tensor().size();
    int m = b.get_tensor().size();
    
    if(n != m)
        return false;

    for(int i=0; i<n; i++){
        if(a.get_tensor()[i] != b.get_tensor()[i])
            return false;
    }
    return true;
}

// USAGE :
// mpirun -n p  main.out <log(n)> <n. of attempts>
int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    using duration = fftcore::Timer::duration;

    assert(argc == 3);

    int s = stoi(argv[1]);
    int attempts = stoi(argv[2]);

    FFTSolver<1> seq_solver(std::make_unique<SequentialFFT<>>());
    FFTSolver<1> mpi_solver(std::make_unique<MPIFFT<>>());
    
    CTensorBase<1> tensorA(1 << s);
    CTensorBase<1> tensorB;
    //operazioni su w
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    
    MPI_Comm_size(MPI_COMM_WORLD, &size);    

    for(int i=0; i<attempts; i++){

        tensorA.get_tensor().setConstant({1.0, -1.0});

        if(rank== 0){
            seq_solver.compute_fft(tensorA,FFT_FORWARD);    
            tensorB = tensorA;
        }

        tensorA.get_tensor().setConstant({1.0, -1.0});

        mpi_solver.compute_fft(tensorA,FFT_FORWARD);

        
        if(rank == 0){
            
            assert(tensor_equality(tensorA, tensorB));
        }
    }
    if(rank==0){
        auto s = seq_solver.get_timer().get_min();
        auto m = mpi_solver.get_timer().get_min();
        cout << "Best Seq: " << s <<endl;
        cout << "Best MPI: " << m << endl;
        cout << "Speedup: " << s/m << endl;
    }

    MPI_Finalize();

}