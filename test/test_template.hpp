#ifndef TEST_FFT_HPP
#define TEST_FFT_HPP

#include <iostream>
#include "../ffft/include/ffft.hpp"

template <class FFTStrategy1, class FFTStrategy2>
void test_fft(char *argv[]){

    using FloatingType1 = typename FFTStrategy1::FloatTypeAlias;
    using FloatingType2 = typename FFTStrategy2::FloatTypeAlias;

    int x = atoi(argv[1]);
    int n = 1 << x;
    std::cout << x << ",";

    FFTSolver<1, FloatingType1> solver1(std::make_unique<FFTStrategy1>());
    FFTSolver<1, FloatingType2> solver2(std::make_unique<FFTStrategy2>());

    CTensorBase<1, FloatingType1> tensor1(n);
    tensor1.get_tensor().setRandom();

    CTensorBase<1, FloatingType2> tensor2(n);
    tensor2.get_tensor() = tensor1.get_tensor().template cast<std::complex<FloatingType2>>(); // cast to different floating type

    solver1.compute_fft(tensor1, FFT_FORWARD);
    double f1 = solver1.get_timer().get_last();
    solver1.get_timer().print_last_formatted();
    std::cout<<",";
    solver1.compute_fft(tensor1, FFT_INVERSE);
    double i1 = solver1.get_timer().get_last();
    solver1.get_timer().print_last_formatted();
    std::cout<<",";

    solver2.compute_fft(tensor2, FFT_FORWARD);
    double f2 = solver2.get_timer().get_last();
    solver2.get_timer().print_last_formatted();
    std::cout<<",";
    solver2.compute_fft(tensor2, FFT_INVERSE);
    double i2 = solver2.get_timer().get_last();
    solver2.get_timer().print_last_formatted();
    std::cout<<",";

    // print speedup
    std::cout<<f2/f1<<","<<i2/i1<<",";

    std::cout << (tensor1.get_tensor().abs() - tensor2.get_tensor().abs()).sum() << std::endl;

    return;
}

#endif //TEST_FFT_HPP
