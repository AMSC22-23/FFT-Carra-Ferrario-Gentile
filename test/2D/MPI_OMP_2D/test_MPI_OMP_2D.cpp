#include <iostream>
#include <memory>
#include "ffft/fftcore.hpp"

using namespace fftcore;

int main(int argc, char** argv)
{
    
    /*MPI initialization*/
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize the dimension of the tensors with the specified dimensions
    assert( argc == (3) && "Usage: ./test_strategy_XD dim1, ..., dimX" ); 
    Eigen::array<Eigen::Index, 2> dimensions;

    // Calculate total elements and print selected dimension sizes
    TensorIdx tmp = 0;
    for(int i = 0; i < 2; i++ )
    {
        if(rank == 0)
            std::cout << argv[i+1] << ","; 
        tmp = 1 << atoi(argv[i+1]);
        dimensions[i] = tmp;
    }
    
    double f2;

    Eigen::Tensor<double, 0> error_inverse;
    Eigen::Tensor<double, 0> error_forward;

    CTensorBase<2, double> tensor(dimensions);
    CTensorBase<2, double> tensor_baseline(dimensions); 

    // Set the random values on process 0
    if(rank == 0){
        tensor.get_tensor().setRandom();
        tensor_baseline.get_tensor()=tensor.get_tensor();
    }

    FFTSolver<2, double> solver_MPI_OMP(std::make_unique<MPI_OMP_FFT_2D<double>>());
    FFTSolver<2, double> solver_baseline(std::make_unique<SequentialFFT_2D<double>>());


    /*
     * FORWARD 
     */    
    solver_MPI_OMP.compute_fft(tensor, FFT_FORWARD);
    double f1 = solver_MPI_OMP.get_timer().get_last();
    if(rank == 0){
        solver_MPI_OMP.get_timer().print_last_formatted();
        std::cout<<",";
        
        // Forward baseline
        solver_baseline.compute_fft(tensor_baseline, FFT_FORWARD);
        f2 = solver_baseline.get_timer().get_last();
        solver_baseline.get_timer().print_last_formatted();
        std::cout<<",";
        
        // norm inf of the error after the transformation
        error_forward = (tensor.get_tensor().abs() - tensor_baseline.get_tensor().abs()).maximum();

        // norm L1
        //error_forward = (tensor.get_tensor().abs() - tensor_baseline.get_tensor().abs()).sum();

    }

    /*
     * INVERSE
     */

    solver_MPI_OMP.compute_fft(tensor, FFT_INVERSE);
    double i1 = solver_MPI_OMP.get_timer().get_last();

    if(rank == 0){
        solver_MPI_OMP.get_timer().print_last_formatted();
        std::cout<<",";
    }

    // Solver 2 
    if(rank == 0){

        // Inverse baseline
        solver_baseline.compute_fft(tensor_baseline, FFT_INVERSE);
        double i2 = solver_baseline.get_timer().get_last();
        solver_baseline.get_timer().print_last_formatted();
        std::cout<<",";

        // print speedup
        std::cout<<f2/f1<<","<<i2/i1<<",";

        
        // norm inf of the error after the inverse
        error_inverse = (tensor.get_tensor().abs() - tensor_baseline.get_tensor().abs()).maximum();

        // norm L1
        //error_inverse = (tensor.get_tensor().abs() - tensor_baseline.get_tensor().abs()).sum();

        // print error inverse 
        std::cout << error_forward << ",";
        std::cout << error_inverse << std::endl;
    }

    MPI_Finalize();
}
