#ifndef TEST_TEMPLATE_HPP
#define TEST_TEMPLATE_HPP

#include <iostream>
#include "ffft/fftcore.hpp"

using namespace fftcore;

/**
 * @brief A templeted function to compare two different strategies. The method prints formatted results to be tabulated with benchmark.sh.
 * @author Lorenzo Gentile
*/
template <int dim, class FFTStrategy1, class FFTStrategy2>
void test_fft(int arcg, char *argv[]){

    using FloatingType1 = typename FFTStrategy1::FloatTypeAlias;
    using FloatingType2 = typename FFTStrategy2::FloatTypeAlias;

    FFTSolver<dim, FloatingType1> solver1(std::make_unique<FFTStrategy1>());
    FFTSolver<dim, FloatingType2> solver2(std::make_unique<FFTStrategy2>());

    Eigen::array<Eigen::Index, dim> dimensions;

    for(int i = 0; i < dim; i++ )
    {
        std::cout << argv[i+1] << ","; 
        dimensions[i] = 1 << atoi(argv[i+1]);
    }

    CTensorBase<dim, FloatingType1> tensor1(dimensions);
    tensor1.get_tensor().setRandom();

    CTensorBase<dim, FloatingType2> tensor2(dimensions);
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

#endif //TEST_TEMPLATE_HPP