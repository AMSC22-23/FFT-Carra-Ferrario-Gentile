#include "../test_template.hpp"


int main(int argc, char *argv[]){

    test_fft<CudaStockhamFFT<double>, fftwFFT<double>>(argv);

}
