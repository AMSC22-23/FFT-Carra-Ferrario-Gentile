#include "../../test_template.hpp"

int main(int argc, char** argv) {
    test_fft<1, MPIFFT<>, SequentialFFT<>>(argc, argv);
}
