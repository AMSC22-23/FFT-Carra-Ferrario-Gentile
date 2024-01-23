#include "../../test_template.hpp"

int main(int argc, char** argv)
{
    test_fft<3, MPIFFT_3D<double>, SequentialFFT_3D<double>>(argc, argv); 
}
