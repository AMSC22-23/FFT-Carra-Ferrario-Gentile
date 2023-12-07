#ifndef MPIFFT_HPP
#define MPIFFT_HPP

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
            void customFFT(CTensor_1D&, FFTDirection) const;

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
 * @author: Daniele Ferrario
 * //@TODO: Works only with double, use MPI scatter and gather, remove workarounds
*/
template<typename FloatingType>
void MPIFFT<FloatingType>::fft(CTensor_1D& input_output, fftcore::FFTDirection fftDirection) const {

    // Utility
    using Complex = std::complex<FloatingType>;
    
    // MPI Infos
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Tensor infos
    int n = input_output.size();
    int log2n = std::log2(n);
    int log2p = std::log2(size);

    // Assertions
    // Add p constraints
    //assert(size<=(std::log2(n)-1) && "Process number must be less or equal than log(n)/2 for optimal results.");
    assert(!(n & (n - 1)) && "FFT length must be a power of 2.");
    // -------------------------------------


    FFTUtils::bit_reversal_permutation(input_output);
    //conjugate if inverse
    if(fftDirection == fftcore::FFT_INVERSE){
        FFTUtils::conjugate(input_output);
    }

    // Create local tensors    
    int local_tensor_size = input_output.size()/size;
    CTensor_1D local_tensor(local_tensor_size);
    
    int starting_local_index = rank*local_tensor_size;
    int i=starting_local_index;


    for(int k=0; k<local_tensor_size; k++){


        local_tensor(k) = input_output(i);
        i++; 
    }
    

    // Bit-reversal permutation 
    // @TODO: This is a workaround 
    FFTUtils::bit_reversal_permutation(local_tensor);

    // // Run fft on local tensors
    SequentialFFT<FloatingType> sequentialFFT;
    sequentialFFT.fft(local_tensor, FFT_FORWARD);
    //noReverseFFT(local_tensor, fftDirection);

    auto *input_output_data = input_output.data();


    if(rank>0){    
        MPI_Send(local_tensor.data(), local_tensor_size, MPI_C_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);

    }else{
        // For Rank 0
        memcpy(input_output.data(), local_tensor.data(), local_tensor_size*sizeof(std::complex<FloatingType>));

        // Overwrite content from other processes
        
        for(int r=1; r<size; r++){
            auto *pos_to_overwrite = input_output_data + r*local_tensor_size;
            MPI_Recv(pos_to_overwrite, local_tensor_size, MPI_C_DOUBLE_COMPLEX, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Now there remains log2(num of processes) steps to do
        Complex w, wm, t, u;
        int m, m2;
        // Cooley-Tukey iterative FFT
        for (int s = log2n - log2p + 1; s <= log2n; ++s)
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

        //re-conjugate and scale if inverse
        if(fftDirection == fftcore::FFT_INVERSE){
            FFTUtils::conjugate(input_output);
            input_output = input_output * Complex(1.0 / n, 0);
        }
        
    }
    MPI_Bcast(input_output.data(), input_output.size(), MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

}



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
