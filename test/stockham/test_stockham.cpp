#include <iostream>
#include "ffft.hpp"

int main(int arcg, char **argv)
{
    int x = atoi(argv[1]);
    int n = 1 << x;
    std::cout << x << ",";

    FFTSolver<1> stockham_solver(std::make_unique<StockhamFFT<>>());
    FFTSolver<1> sequential_solver(std::make_unique<SequentialFFT<>>());

    CTensorBase<1> tensor_sequential(n);
    tensor_sequential.get_tensor().setRandom();

    CTensorBase<1> tensor_stockham(tensor_sequential);

    stockham_solver.compute_fft(tensor_stockham, FFT_FORWARD);
    double stockham_f = stockham_solver.get_timer().get_last();
    stockham_solver.get_timer().print_last_formatted();
    std::cout<<",";
    stockham_solver.compute_fft(tensor_stockham, FFT_INVERSE);
    double stockham_i = stockham_solver.get_timer().get_last();
    stockham_solver.get_timer().print_last_formatted();
    std::cout<<",";

    sequential_solver.compute_fft(tensor_sequential, FFT_FORWARD);
    double seq_f = sequential_solver.get_timer().get_last();
    sequential_solver.get_timer().print_last_formatted();
    std::cout<<",";
    sequential_solver.compute_fft(tensor_sequential, FFT_INVERSE);
    double seq_i = sequential_solver.get_timer().get_last();
    sequential_solver.get_timer().print_last_formatted();
    std::cout<<",";

    // print speedup
    std::cout<<seq_f/stockham_f<<","<<seq_i/stockham_i<<",";

    std::cout << (tensor_stockham.get_tensor().abs() - tensor_sequential.get_tensor().abs()).sum() << std::endl;
}
