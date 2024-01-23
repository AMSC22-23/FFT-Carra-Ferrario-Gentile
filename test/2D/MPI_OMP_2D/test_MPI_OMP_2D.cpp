#include <iostream>
#include <memory>
#include "ffft/fftcore.hpp"

using namespace fftcore;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize the dimension of the tensors with the specified dimensions
    assert( argc == (3) && "Usage: ./test_strategy_XD dim1, ..., dimX" ); 
    Eigen::array<Eigen::Index, 2> dimensions;

    // Calculate total elements and print selected dimension sizes
    int total_elements_number = 0, tmp = 0;
    for(int i = 0; i < 2; i++ )
    {
        if(rank == 0)
            std::cout << argv[i+1] << ","; 
        tmp = 1 << atoi(argv[i+1]);
        total_elements_number += tmp;
        dimensions[i] = tmp;
    }
    
    CTensorBase<2, double> tensor1(dimensions);
    CTensorBase<2, double> tensor2(dimensions); 

    // Set the random values on process 0
    if(rank == 0){
        tensor1.get_tensor().setRandom();
        tensor2.get_tensor()=tensor1.get_tensor();
    }


    FFTSolver<2, double> solverMPI(std::make_unique<MPI_OMP_FFT_2D<double>>());
    //FFTSolver<2, double> solver2(std::make_unique<OmpFFT_2D<double>>());
    FFTSolver<2, double> solver2(std::make_unique<SequentialFFT_2D<double>>());

    solverMPI.compute_fft(tensor1, FFT_FORWARD);
    double f1 = solverMPI.get_timer().get_last();
    if(rank == 0){
        solverMPI.get_timer().print_last_formatted();
        std::cout<<",";
    }


    solverMPI.compute_fft(tensor1, FFT_INVERSE);
    double i1 = solverMPI.get_timer().get_last();


    if(rank == 0){
        solverMPI.get_timer().print_last_formatted();
        std::cout<<",";
    }

    // Solver 2 
    if(rank == 0){

        // Forward
        solver2.compute_fft(tensor2, FFT_FORWARD);
        double f2 = solver2.get_timer().get_last();
        solver2.get_timer().print_last_formatted();
        std::cout<<",";

        // Inverse
        solver2.compute_fft(tensor2, FFT_INVERSE);
        double i2 = solver2.get_timer().get_last();
        solver2.get_timer().print_last_formatted();
        std::cout<<",";

        // print speedup
        std::cout<<f2/f1<<","<<i2/i1<<",";

        
        std::cout << (tensor1.get_tensor().abs() - tensor2.get_tensor().abs()).sum() << std::endl;
    }

    MPI_Finalize();
}
