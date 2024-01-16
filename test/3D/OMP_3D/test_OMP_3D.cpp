#include <iostream>
#include <memory>
#include "ffft.hpp"
#include "../../test_template.hpp"
using namespace fftcore;

int main(int argc, char** argv)
{
    test_fft<3, OmpFFT_3D<double>, SequentialFFT_3D<double>>(argc, argv); 
}
