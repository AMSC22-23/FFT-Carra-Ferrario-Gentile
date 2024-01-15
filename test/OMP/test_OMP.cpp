#include <iostream>
#include "ffft.hpp"
#include "../test_template.hpp"

int main(int argc, char** argv){

        test_fft<1, OmpFFT<>, SequentialFFT<>>(argc, argv);
        /*
        int x = atoi(argv[1]);
        int n = 1 << x;
        std::cout<<x<<",";
        
        FFTSolver<1> omp_solver(std::make_unique<OmpFFT<>>());
        FFTSolver<1> sequential_solver(std::make_unique<SequentialFFT<>>());
        
        CTensorBase<1> tensor_sequential(n);
        tensor_sequential.get_tensor().setRandom();
        
        CTensorBase<1> tensor_omp(tensor_sequential);


        omp_solver.compute_fft(tensor_omp, FFT_FORWARD);
        double omp_f = omp_solver.get_timer().get_last();
        omp_solver.get_timer().print_last_formatted();
        std::cout<<",";
        omp_solver.compute_fft(tensor_omp, FFT_INVERSE);
        double omp_i = omp_solver.get_timer().get_last();
        omp_solver.get_timer().print_last_formatted();
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
        std::cout<<seq_f/omp_f<<","<<seq_i/omp_i<<",";

        std::cout << (tensor_omp.get_tensor().abs() - tensor_sequential.get_tensor().abs()).sum() << std::endl;
        */
}
