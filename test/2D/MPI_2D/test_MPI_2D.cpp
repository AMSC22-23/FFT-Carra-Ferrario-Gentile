#include "../../test_template_MPI.hpp"


// USAGE :
// mpirun -n <p>  main.out <log(n)> <n. of attempts>
int main(int argc, char **argv){
    test_fft_mpi<2, MPIFFT_2D<>, SequentialFFT_2D<>>(argc, argv); 
}