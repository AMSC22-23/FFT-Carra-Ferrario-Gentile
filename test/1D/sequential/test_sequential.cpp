#include <iostream>
#include <memory>
#include "ffft.hpp"
#include "../../test_template.hpp"

using namespace fftcore;

int main(int argc, char **argv){
   test_fft<1, SequentialFFT<double>, fftwFFT<double>>(argc, argv); 
   //test_fft<2, SequentialFFT_2D<double>, fftwFFT<double>>(argv); 
   //test_fft<3, SequentialFFT<>, fftwFFT<>>(argv); 

}