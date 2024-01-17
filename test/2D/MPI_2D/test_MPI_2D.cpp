
#include "ffft.hpp"
#include <memory>
#include <cmath>
#include "../../test_template.hpp"

using namespace fftcore;



// USAGE :
// mpirun -n <p>  main.out <log(n)> <n. of attempts>
int main(int argc, char **argv){
    test_fft<2, MPIFFT_2D<>, SequentialFFT_2D<>>(argc, argv); 
}