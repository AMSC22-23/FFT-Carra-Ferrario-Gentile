#include "../../test_template.hpp"

int main(int argc, char **argv)
{
    test_fft<1, StockhamFFT<double>, fftwFFT<double>>(argc, argv); 
}