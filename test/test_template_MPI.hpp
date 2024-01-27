#ifndef TEST_TEMPLATE_MPI_HPP
#define TEST_TEMPLATE_MPI_HPP

#include <iostream>
#include <vector>
#include "ffft/fftcore.hpp"

/**
 * @brief A templeted function to compare an MPI strategy with another strategy.
 * @note The MPI strategy has to be the first argument.
 * @author Daniele Ferrario
*/
using namespace fftcore;

template <int dim, class FFTStrategy1, class FFTStrategy2>
void test_fft_mpi(int argc, char *argv[]){

    if(argc != dim + 1)
    {
        std::cout << "Usage: " << argv[0] << " <dim1> <dim2> ... <dimN>" << std::endl;
        return;
    }

    using FloatingType1 = typename FFTStrategy1::FloatTypeAlias;
    using FloatingType2 = typename FFTStrategy2::FloatTypeAlias;


    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);    
    MPI_Datatype mpi_datatype1 = std::is_same<FloatingType1, double>::value ? MPI_C_DOUBLE_COMPLEX : MPI_C_FLOAT_COMPLEX;

    Eigen::Tensor<double, 0> error_inverse;
    Eigen::Tensor<double, 0> error_forward;

    FFTSolver<dim, FloatingType1> solver(std::make_unique<FFTStrategy1>());
    FFTSolver<dim, FloatingType2> solver_baseline(std::make_unique<FFTStrategy2>());

    // Initialize the dimension of the tensors with the specified dimensions
    assert( argc == (dim + 1) && "Usage: ./test_strategy_XD dim1, ..., dimX" ); 
    Eigen::array<Eigen::Index, dim> dimensions;

    // Calculate total elements and print selected dimension sizes
    int total_elements_number = 1, tmp = 0;
    for(int i = 0; i < dim; i++ )
    {
        if(rank == 0)
            std::cout << argv[i+1] << ","; 
        tmp = 1 << atoi(argv[i+1]);
        total_elements_number *= tmp;
        dimensions[i] = tmp;
    }

    double f2; 

    CTensorBase<dim, FloatingType1> tensor(dimensions);
    CTensorBase<dim, FloatingType2> tensor_baseline(dimensions);    


    // Set the random values on process 0
    if(rank == 0)
        tensor.get_tensor().setRandom();

    // And copy it on the other processes
    MPI_Bcast(tensor.get_tensor().data(), total_elements_number, mpi_datatype1, 0, MPI_COMM_WORLD);

    // Save the copy for the second solver
        if( rank == 0 ){
        tensor_baseline.get_tensor() = tensor.get_tensor().template cast<std::complex<FloatingType2>>(); // cast to different floating type
    }

    // SOLVER 1 - Use MPI Solver here
    
    // Forward
    solver.compute_fft(tensor, FFT_FORWARD);
    double f1 = solver.get_timer().get_last();
    if(rank == 0){
        solver.get_timer().print_last_formatted();
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

    // Inverse
    MPI_Bcast(tensor.get_tensor().data(), total_elements_number, mpi_datatype1, 0, MPI_COMM_WORLD);
    solver.compute_fft(tensor, FFT_INVERSE);
    double i1 = solver.get_timer().get_last();

    if(rank == 0){
        solver.get_timer().print_last_formatted();
        std::cout<<",";
    }

    // Solver 2 
    if(rank == 0){

        // Inverse
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
    return;
}

#endif //TEST_TEMPLATE_MPI_HPP
