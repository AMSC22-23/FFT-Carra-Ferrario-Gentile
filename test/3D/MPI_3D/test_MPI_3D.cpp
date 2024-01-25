#include "../../test_template_MPI.hpp"

int main(int argc, char** argv)
{
    test_fft_mpi<3, MPIFFT_3D<double>, SequentialFFT_3D<double>>(argc, argv); 
}
