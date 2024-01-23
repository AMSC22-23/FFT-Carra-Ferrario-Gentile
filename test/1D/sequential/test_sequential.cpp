#include "../../test_template.hpp"

int main(int argc, char **argv){
   test_fft<1, SequentialFFT<double>, fftwFFT<double>>(argc, argv); 
}