#include "../../ffft/src/fftcore/Tensor/TensorFFTBase.hpp"
#include "../../ffft/src/fftcore/Strategy/SequentialFFT/SequentialFFT.hpp"
#include "../../ffft/src/fftcore/Strategy/MPIFFT/MPIFFT.hpp"
#include "../../ffft/src/fftcore/FFTSolver.hpp"

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

template<typename FloatingType>
void random_tensor(const int real_bound_mod, const int imag_bound_mod, CTensorBase<1> &tensor){
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> real_distribution(-real_bound_mod, real_bound_mod);
    std::uniform_int_distribution<int> imag_distribution(-imag_bound_mod, imag_bound_mod);


    for(int i=0; i<tensor.get_tensor().size(); i++){
            // Generate random real and imaginary parts
        FloatingType random_real = real_distribution(gen);
        FloatingType random_imag = imag_distribution(gen);

        tensor.get_tensor()[i] = std::complex<FloatingType>({random_real, random_imag});
    }

}
// USAGE :
// mpirun -n <p>  main.out <log(n)> <n. of attempts>
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
        
        
        if(rank==0){
            /*
           tensorA.get_tensor().setValues({
            {1.0, 0.0},
            {2.0, 0.0},
            {4.0, 0.0},
            {3.0, 0.0},

           });*/
           tensorA.get_tensor().setRandom();
           
        }
       
        MPI_Bcast(tensorA.get_tensor().data(), tensorA.get_tensor().size(), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

        //cout << "ref: " << tensorReference.get_tensor()<< endl<< endl;
        //tensorA.get_tensor().setConstant({1.0, -1.0});
        tensorB = tensorA;

        if(rank== 0){
            seq_solver.compute_fft(tensorA,FFT_FORWARD);    
        }

        //tensorA.get_tensor().setConstant({1.0, -1.0});

        mpi_solver.compute_fft(tensorB,FFT_FORWARD);

        
        if(rank == 0){
            
            assert(tensor_equality(tensorA, tensorB));
        }
    }
    if(rank==0){
        auto seq_min = seq_solver.get_timer().get_min();
        auto mpi_min = mpi_solver.get_timer().get_min();
        //cout << tensorA.get_tensor() << endl;
        //cout << tensorB.get_tensor() << endl;
        cout << "Best Seq: " << seq_min <<endl;
        cout << "Best MPI: " << mpi_min << endl;
        cout << "Speedup: " << seq_min/mpi_min << endl;

    }

    MPI_Finalize();

}