#include "../../test_template.hpp"


// USAGE :
// mpirun -n <p>  main.out <log(n)> <n. of attempts>
int main(int argc, char **argv){
    test_fft<2, MPIFFT_2D<>, SequentialFFT_2D<>>(argc, argv); 
}