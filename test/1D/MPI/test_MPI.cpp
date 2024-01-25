#include "../../test_template_MPI.hpp"
int main(int argc, char** argv) {
    test_fft_mpi<1, MPIFFT<>, SequentialFFT<>>(argc, argv);
}
