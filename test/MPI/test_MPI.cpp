#define pout if(rank == 0) std::cout

#include <iostream>
#include "ffft.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int x = atoi(argv[1]);
    int n = 1 << x;
    pout << x << ",";

    FFTSolver<1> mpi_solver(std::make_unique<MPIFFT<>>());
    FFTSolver<1> sequential_solver(std::make_unique<SequentialFFT<>>());

    CTensorBase<1> tensor_sequential(n);

    if(rank == 0)
        tensor_sequential.get_tensor().setRandom();

    MPI_Bcast(tensor_sequential.get_tensor().data(), n, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    CTensorBase<1> tensor_mpi(tensor_sequential);

    // Forward MPI
    mpi_solver.compute_fft(tensor_mpi, FFT_FORWARD);
    double mpi_f = mpi_solver.get_timer().get_last();
    if(rank == 0)
        mpi_solver.get_timer().print_last_formatted();
    pout << ",";

    // Inverse MPI
    MPI_Bcast(tensor_mpi.get_tensor().data(), n, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    mpi_solver.compute_fft(tensor_mpi, FFT_INVERSE);
    double mpi_i = mpi_solver.get_timer().get_last();
    if(rank == 0)
        mpi_solver.get_timer().print_last_formatted();
    pout << ",";

    if(rank == 0){
        // Forward sequential
        sequential_solver.compute_fft(tensor_sequential, FFT_FORWARD);
        double seq_f = sequential_solver.get_timer().get_last();
        sequential_solver.get_timer().print_last_formatted();
        pout << ",";

        // Inverse sequential
        sequential_solver.compute_fft(tensor_sequential, FFT_INVERSE);
        double seq_i = sequential_solver.get_timer().get_last();
        sequential_solver.get_timer().print_last_formatted();
        pout << ",";
    
        // Speedup amd error
        pout << seq_f / mpi_f << "," << seq_i / mpi_i << ",";
        pout << (tensor_mpi.get_tensor().abs() - tensor_sequential.get_tensor().abs()).sum() << "\n";
    }

    MPI_Finalize();
}
