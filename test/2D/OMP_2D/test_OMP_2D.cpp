#include "../../test_template.hpp"

int main(int argc, char** argv)
{
    test_fft<2, OmpFFT_2D<double>, SequentialFFT_2D<double>>(argc, argv); 
}
