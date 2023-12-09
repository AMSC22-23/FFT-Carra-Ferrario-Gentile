#ifndef MPIFFT_HPP
#define MPIFFT_HPP

#define INIT_TIME_MEASURE double start, end, check;

#define MEASURE_TIME_START start = MPI_Wtime();

#define MEASURE_TIME_END(message) \
    do { \
        end = MPI_Wtime(); \
        if(rank==0){\
            std::cout << "p: " << rank << " " << message << ": " << (end - start)*1.0e6 << " microseconds" << std::endl; \
            check += (end - start)*1.0e6; \
        } \
    } while(0);

#define CHECK_TIME cout << "Check mpi time: " << check << endl;

#include "../../FFTSolver.hpp"
#include "../../utils/FFTUtils.hpp"
#include "../SequentialFFT/SequentialFFT.hpp"
#include <iostream>
#include <mpi.h>

using namespace fftcore;
using namespace std;

template<typename FloatingType = double>
class MPIFFT:
public FFT_1D<FloatingType>,
public FFT_2D<FloatingType>, 
public FFT_3D<FloatingType>{
		public:
            using typename FFT_1D<FloatingType>::RTensor_1D;
            using typename FFT_1D<FloatingType>::CTensor_1D;

            using typename FFT_2D<FloatingType>::RTensor_2D;
            using typename FFT_2D<FloatingType>::CTensor_2D;

            using typename FFT_3D<FloatingType>::RTensor_3D;
            using typename FFT_3D<FloatingType>::CTensor_3D;

			void fft(const CTensor_1D& , CTensor_1D&, FFTDirection) const;
			
			void fft(const RTensor_1D&, CTensor_1D&, FFTDirection) const;

			void fft(CTensor_1D&, FFTDirection) const;

			void fft(const CTensor_2D&, CTensor_2D&, FFTDirection) const;

			void fft(const RTensor_2D&, CTensor_2D&, FFTDirection) const;

			void fft(CTensor_2D&, FFTDirection) const;

			void fft(const CTensor_3D&, CTensor_3D&, FFTDirection) const ;

			void fft(const RTensor_3D&, CTensor_3D&, FFTDirection) const ;

			void fft(CTensor_3D&, FFTDirection) const ;

			~MPIFFT() = default;
        private :
            void _fft_no_reverse_no_conjugate(CTensor_1D&, int) const;

};


template<typename FloatingType>
void MPIFFT<FloatingType>::fft(const CTensor_1D& input, CTensor_1D& output, FFTDirection) const {
    std::cout<<"fft 1-d C-C out-of-place"<<std::endl;

};

template<typename FloatingType>
void MPIFFT<FloatingType>::fft(const RTensor_1D&, CTensor_1D&, FFTDirection) const {
    std::cout<<"fft 1-d R-C out-of-place"<<std::endl;
};


/**
 * In place FFT using MPI parallelization. Every process will compute a 
 * sub tree of the iterational Cooley-tuekey algorithm, until there will 
 * occure dependency between them. Then, only one process will complete the
 * remaining steps of the algoritm, precisely doing log(p_size) steps with p_size
 * being the number of available processes.
 * 
 * @author: Daniele Ferrario
 * @TODO: avoiding full copy of original tensor in every process?
 * @TODO: add exceptions instead of assertions
*/

template<typename FloatingType>
void MPIFFT<FloatingType>::fft(CTensor_1D& input_output, fftcore::FFTDirection fftDirection) const {

    INIT_TIME_MEASURE

    // Utility
    using Complex = std::complex<FloatingType>;
    
    // MPI Infos
    int rank, size;
    MPI_Datatype mpi_datatype = std::is_same<FloatingType, double>::value ? MPI_C_DOUBLE_COMPLEX : MPI_C_FLOAT_COMPLEX;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Tensor infos
    int n = input_output.size();
    int log2n = std::log2(n);
    double log2p = std::log2(size);

    // Assertions
    // Add p constraints
    assert(!(n & (n - 1)) && "FFT length must be a power of 2.");
    assert((size <= n/2 && std::ceil(log2p)==std::floor(log2p)) && "Process number must be a power of two and less or equal than n/2).");
    // -------------------------------------

    MEASURE_TIME_START
    // Create local tensors    
    int local_tensor_size = input_output.size()/size;
    MEASURE_TIME_END("Global array permutation")
    
    int starting_local_index = rank*local_tensor_size;
    int i=starting_local_index;


    //c Conjugate input if ifft
    if(fftDirection == fftcore::FFT_INVERSE){
        FFTUtils::conjugate(input_output);
    }


    // Create local tensors
    MEASURE_TIME_START
    CTensor_1D local_tensor(local_tensor_size);
    for(int k=0; k<local_tensor_size; k++){
        local_tensor(k) = input_output(i);
        i++; 
    }
    MEASURE_TIME_END("Local tensors creation")


    MEASURE_TIME_START
    // Run fft on local tensors ( subtrees )
    MPIFFT::_fft_no_reverse_no_conjugate(local_tensor, 1);
    MEASURE_TIME_END("Local array FFT")

    // Reconstruct the original tenso gathering local tensors
    MPI_Gather(local_tensor.data(), local_tensor_size, mpi_datatype, input_output.data(), local_tensor_size, mpi_datatype, 0, MPI_COMM_WORLD);
    
    if(rank == 0){

        // Now there remains log2(num of processes) steps to do
        MEASURE_TIME_START
        MPIFFT::_fft_no_reverse_no_conjugate(input_output, log2n - log2p + 1);
        MEASURE_TIME_END("Last global FFT")

        //re-conjugate and scale if inverse
        if(fftDirection == fftcore::FFT_INVERSE){
            FFTUtils::conjugate(input_output);
            input_output = input_output * Complex(1.0 / n, 0);
        }
        
    }

    // Copy global tensor in all processes
    MPI_Bcast(input_output.data(), input_output.size(), mpi_datatype, 0, MPI_COMM_WORLD);

}

template<typename FloatingType>
void MPIFFT<FloatingType>::_fft_no_reverse_no_conjugate(CTensor_1D& input_output, int starting_depth) const{
        using Complex = std::complex<FloatingType>;
        
        int n = input_output.size();
        assert(!(n & (n - 1)) && "FFT length must be a power of 2.");

        int log2n = std::log2(n);

        Complex w, wm, t, u;
        int m, m2;
        // Cooley-Tukey iterative FFT
        for (int s = starting_depth; s <= log2n; ++s)
        {
            m = 1 << s;  // 2 power s
            m2 = m >> 1; // m2 = m/2 -1
            wm = exp(Complex(0, -2 * M_PI / m)); // w_m = e^(-2*pi/m)

            for(int k = 0; k < n; k += m)
            {
                w = Complex(1, 0);
                for(int j = 0; j < m2; ++j)
                {
                    t = w * input_output[k + j + m2];
                    u = input_output[k + j];

                    input_output[k + j] = u + t;
                    input_output[k + j + m2] = u - t;

                    w *= wm;
                }
            }
        }

};

template<typename FloatingType>
void MPIFFT<FloatingType>::fft(const CTensor_2D&, CTensor_2D&, FFTDirection) const {
    std::cout<<"fft 2-d C-C out-of-place"<<std::endl;
};

template<typename FloatingType>
void MPIFFT<FloatingType>::fft(const RTensor_2D&, CTensor_2D&, FFTDirection) const {
    std::cout<<"fft 2-d R-C out-of-place"<<std::endl;

};

template<typename FloatingType>
void MPIFFT<FloatingType>::fft(CTensor_2D&, FFTDirection) const {
    std::cout<<"fft 2-d in-place"<<std::endl;
};

template<typename FloatingType>
void MPIFFT<FloatingType>::fft(const CTensor_3D&, CTensor_3D&, FFTDirection) const {
    std::cout<<"fft 3-d C-C out-of-place"<<std::endl;
};

template<typename FloatingType>
void MPIFFT<FloatingType>::fft(const RTensor_3D&, CTensor_3D&, FFTDirection) const {
    std::cout<<"fft 3-d R-C out-of-place"<<std::endl;

};

template<typename FloatingType>
void MPIFFT<FloatingType>::fft(CTensor_3D&, FFTDirection) const {
    std::cout<<"fft 3-d in-place"<<std::endl;
};

#endif //MPIFFT_HPP
