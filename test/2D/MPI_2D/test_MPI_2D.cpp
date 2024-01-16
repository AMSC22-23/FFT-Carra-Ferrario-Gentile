
#include "ffft.hpp"
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
// mpirun -n <p>  main.out <log(n)> <n. of attempts>
int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    using duration = fftcore::Timer::duration;

    assert(argc == 3);

    int s = stoi(argv[1]);
    int attempts = stoi(argv[2]);
    FFTSolver<2> mpi_solver(std::make_unique<MPIFFT_2D<>>());

    CTensorBase<2> tensorA(1 << s, 1 << s);
    CTensorBase<2> tensorB;

    //operazioni su w
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    
    MPI_Comm_size(MPI_COMM_WORLD, &size);    


    for(int i=0; i<attempts; i++){
        
        
        if(rank==0){
           tensorA.get_tensor().setConstant(1);
        }
       
        MPI_Bcast(tensorA.get_tensor().data(), tensorA.get_tensor().size(), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

        tensorB = tensorA;


        mpi_solver.compute_fft(tensorB,FFT_FORWARD);

        
    }
    if(rank==0){
        auto mpi_min = mpi_solver.get_timer().get_min();
        //cout << tensorA.get_tensor() << endl;
        cout << tensorB.get_tensor() << endl;
        cout << "Best MPI: " << mpi_min << endl;
        cout << "Error: " << (tensorA.get_tensor().abs() - tensorB.get_tensor().abs()).sum() << endl;

    }

    MPI_Finalize();

}