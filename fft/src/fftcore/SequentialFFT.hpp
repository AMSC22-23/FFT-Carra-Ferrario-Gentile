#include "FFTSolver.hpp"
#include "utils/FFTUtils.hpp"
#include <iostream>

using namespace fftcore;
using namespace std;

template<typename FloatingType = double>
class SequentialFFT:
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
			~SequentialFFT() = default;
};


template<typename FloatingType>
void SequentialFFT<FloatingType>::fft(const CTensor_1D& input, CTensor_1D& output, FFTDirection) const {
    std::cout<<"fft 1-d C-C out-of-place"<<std::endl;

};

template<typename FloatingType>
void SequentialFFT<FloatingType>::fft(const RTensor_1D&, CTensor_1D&, FFTDirection) const {
    std::cout<<"fft 1-d R-C out-of-place"<<std::endl;
};


/**
 * @author: Lorenzo Gentile
*/
template<typename FloatingType>
void SequentialFFT<FloatingType>::fft(CTensor_1D& input_output, fftcore::FFTDirection fftDirection) const {

    using Complex = std::complex<FloatingType>;
    int n = input_output.size();
    int log2n = std::log2(n);
     
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
    // Cooley-Tukey iterative FFT
    for (int s = 1; s <= log2n; ++s) {
        m = 1 << s;         // 2 power s
        m2 = m >> 1;        // m2 = m/2 -1
        w = Complex(1, 0);
        wm = exp(Complex(0, -2 * M_PI / m)); // w_m = e^(-2*pi/m)

        for (int j = 0; j < m2; ++j) {
            for (int k = j; k < n; k += m) {
                t = w * input_output[k + m2];
                u = input_output[k];
                input_output[k] = u + t;
                input_output[k + m2] = u - t;
            }
            w *= wm;
        }
    }


    if(fftDirection == fftcore::FFTDirection::FFT_INVERSE){
        for(int i=0; i<n; i++){
            input_output[i] /= n;
        }
    }
};

template<typename FloatingType>
void SequentialFFT<FloatingType>::fft(const CTensor_2D&, CTensor_2D&, FFTDirection) const {
    std::cout<<"fft 2-d C-C out-of-place"<<std::endl;
};

template<typename FloatingType>
void SequentialFFT<FloatingType>::fft(const RTensor_2D&, CTensor_2D&, FFTDirection) const {
    std::cout<<"fft 2-d R-C out-of-place"<<std::endl;

};

template<typename FloatingType>
void SequentialFFT<FloatingType>::fft(CTensor_2D&, FFTDirection) const {
    std::cout<<"fft 2-d in-place"<<std::endl;
};

template<typename FloatingType>
void SequentialFFT<FloatingType>::fft(const CTensor_3D&, CTensor_3D&, FFTDirection) const {
    std::cout<<"fft 3-d C-C out-of-place"<<std::endl;
};

template<typename FloatingType>
void SequentialFFT<FloatingType>::fft(const RTensor_3D&, CTensor_3D&, FFTDirection) const {
    std::cout<<"fft 3-d R-C out-of-place"<<std::endl;

};

template<typename FloatingType>
void SequentialFFT<FloatingType>::fft(CTensor_3D&, FFTDirection) const {
    std::cout<<"fft 3-d in-place"<<std::endl;
};
