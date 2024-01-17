#ifndef TEST_FFT_HPP
#define TEST_FFT_HPP

#include <iostream>
#include "../ffft/include/ffft.hpp"
#include <vector>

/**
 * @brief A templeted function to compare two different strategies.
 * @note If using a MPI strategy, only use it as FFTStrategy1.
 * @author Lorenzo Gentile, Daniele Ferrario
*/
template <int dim, class FFTStrategy1, class FFTStrategy2>
void test_fft(int argc, char *argv[]){


    using FloatingType1 = typename FFTStrategy1::FloatTypeAlias;
    using FloatingType2 = typename FFTStrategy2::FloatTypeAlias;


    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);    
    MPI_Datatype mpi_datatype1 = std::is_same<FloatingType1, double>::value ? MPI_C_DOUBLE_COMPLEX : MPI_C_FLOAT_COMPLEX;


    FFTSolver<dim, FloatingType1> solver1(std::make_unique<FFTStrategy1>());
    FFTSolver<dim, FloatingType2> solver2(std::make_unique<FFTStrategy2>());

    // Initialize the dimension of the tensors with the specified dimensions
    assert( argc == (dim + 1) && "Usage: ./test_strategy_XD dim1, ..., dimX" ); 
    Eigen::array<Eigen::Index, dim> dimensions;

    // Calculate total elements and print selected dimension sizes
    int total_elements_number = 0, tmp = 0;
    for(int i = 0; i < dim; i++ )
    {
        if(rank == 0)
            std::cout << argv[i+1] << ","; 
        tmp = 1 << atoi(argv[i+1]);
        total_elements_number += tmp;
        dimensions[i] = tmp;
    }

    

    CTensorBase<dim, FloatingType1> tensor1(dimensions);
    CTensorBase<dim, FloatingType2> tensor2(dimensions);    


    // Set the random values on process 0
    if(rank == 0)
        tensor1.get_tensor().setRandom();

    // And copy it on the other processes
    MPI_Bcast(tensor1.get_tensor().data(), total_elements_number, mpi_datatype1, 0, MPI_COMM_WORLD);

    // Save the copy for the second solver
    if( rank == 0 ){
        tensor2.get_tensor() = tensor1.get_tensor().template cast<std::complex<FloatingType2>>(); // cast to different floating type
    }

    // SOLVER 1 - Use MPI Solver here
    
    // Forward
    solver1.compute_fft(tensor1, FFT_FORWARD);
    double f1 = solver1.get_timer().get_last();
    if(rank == 0){
        solver1.get_timer().print_last_formatted();
        std::cout<<",";
    }

    // Inverse
    MPI_Bcast(tensor1.get_tensor().data(), total_elements_number, mpi_datatype1, 0, MPI_COMM_WORLD);
    solver1.compute_fft(tensor1, FFT_INVERSE);
    double i1 = solver1.get_timer().get_last();

    if(rank == 0){
        solver1.get_timer().print_last_formatted();
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
    return;
}

#endif //TEST_FFT_HPP
