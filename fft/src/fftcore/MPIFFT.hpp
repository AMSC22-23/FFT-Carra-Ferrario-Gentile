
#include "FFTSolver.hpp"
#include "utils/FFTUtils.hpp"
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
 * @author: Lorenzo Gentile, Daniele Ferrario
*/
template<typename FloatingType>
void MPIFFT<FloatingType>::fft(CTensor_1D& input_output, fftcore::FFTDirection fftDirection) const {

    using Complex = std::complex<FloatingType>;
    int n = input_output.size();
    int log2n = std::log2(n);
    
    int pi, total_p;
    MPI_Comm_rank(MPI_COMM_WORLD, &pi);
    MPI_Comm_size(MPI_COMM_WORLD, &total_p);
    
    cout << "Process number: " << pi<< endl;
    //@TODO: check
    assert(pi<=n/2 && "Process number must be less or equal than n/2.");
        
    assert(!(n & (n - 1)) && "FFT length must be a power of 2.");


        // Bit-reversal permutation
        for (unsigned int i = 0; i < n; ++i) {
            unsigned int rev = FFTUtils::reverseBits(i, log2n);
            if (i < rev) {
                std::swap(input_output[i], input_output[rev]);
            }
        }

    Complex w, wm, t, u;
    int m, m2;
    int k1_pi, k2_pi;

    // Cooley-Tukey iterative FFT
    for (int s = 1; s <= log2n; ++s) {
        m = 1 << s;         // 2 power s
        m2 = m >> 1;        // m2 = m/2 -1
        wm = exp(Complex(0, 2 * M_PI / m)); // w_m = e^(2*pi/m)

        // If process index is even
        
        k1_pi = pi;
        k2_pi = 1 << (s-1);

        wm = std::pow(wm, pi);
        t = wm * input_output[k2_pi];
        u = input_output[k1_pi];
        
        input_output[k1_pi] = u + t;
        input_output[k2_pi] = u - t;
        

        MPI_Barrier(MPI_COMM_WORLD);

    }
   
    if(pi == 0 && fftDirection == fftcore::FFTDirection::FFT_INVERSE){
        for(int i=0; i<n; i++){
            input_output[i] /= n;
        }
        
        // Re-oredering
        // @TODO: I don't know if it's correct, but it works (Ferra)
        // Also, no need to conjugate anything apparently
        for (unsigned int i = 1; i < n/2; ++i) {
            std::swap(input_output[i], input_output[n-i]);
        }
        

    }
    MPI_Barrier(MPI_COMM_WORLD);

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
