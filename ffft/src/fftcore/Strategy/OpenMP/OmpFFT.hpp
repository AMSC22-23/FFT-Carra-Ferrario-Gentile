#ifndef OMPFFT_HPP
#define OMPFFT_HPP

#include <iostream>
#include <omp.h>
#include <cmath>
#include "../../FFTSolver.hpp"
#include "../../utils/FFTUtils.hpp"

using namespace std;
namespace fftcore{

    template<typename FloatingType = double>
    class OmpFFT:
    public FFT_1D<FloatingType>{
            public:
                using typename FFT_1D<FloatingType>::RTensor_1D;
                using typename FFT_1D<FloatingType>::CTensor_1D;


                void fft(const CTensor_1D& , CTensor_1D&, FFTDirection) const;
                
                void fft(const RTensor_1D&, CTensor_1D&, FFTDirection) const;

                void fft(CTensor_1D&, FFTDirection) const;

                ~OmpFFT() = default;
    };


    template<typename FloatingType>
    void OmpFFT<FloatingType>::fft(const CTensor_1D& input, CTensor_1D& output, FFTDirection) const {
        throw NotSupportedException("Operation is not supported");
    };

    template<typename FloatingType>
    void OmpFFT<FloatingType>::fft(const RTensor_1D&, CTensor_1D&, FFTDirection) const {
        throw NotSupportedException("Operation is not supported");
    };

    /**
     * @author: Edoardo Carra
    */
    template<typename FloatingType>
    void OmpFFT<FloatingType>::fft(CTensor_1D& input_output, FFTDirection fftDirection) const {

        using Complex = std::complex<FloatingType>;
        int n = input_output.size();
        Complex w, wm, t, u;
        int m, m2; 
		int log2n = std::log2(n);
		int rev;
        assert(!(n & (n - 1)) && "FFT length must be a power of 2.");

        //conjugate if inverse
        if(fftDirection == FFT_INVERSE){
            FFTUtils::conjugate(input_output);
        }

        // Bit-reversal permutation
		#pragma omp parallel private(m, m2, w, t, u, wm, rev)
		{
		#pragma omp for
        for (int i = 0; i < input_output.size(); ++i)
        {
            rev = FFTUtils::reverseBits(i, log2n);
            if (i < rev)
            {
                std::swap(input_output[i], input_output[rev]);
            }
        }

		#pragma omp barrier

        // Cooley-Tukey iterative FFT	
        for (int s = 1; s <= log2n; ++s) {
            m = 1 << s;         // 2 power s
            m2 = m >> 1;        // m2 = m/2 -1
			wm = exp(Complex(0,-2*M_PI/m));

			#pragma omp for 
            for (int k = 0; k < n; k += m) {
				w=Complex(1,0);		    
//              #pragma omp parallel for private(w,t,u)
				#pragma omp simd
				for (int j = 0; j < m2; ++j) {
                    t = w * input_output[j + k + m2];
                    u = input_output[j + k];
                    
                    input_output[k+j] = u + t;
                    input_output[k + m2 + j] = u - t;
					w=w*wm;
                }
            }
        }
		}
       //re-conjugate and scale if inverse
        if(fftDirection == FFT_INVERSE){
            FFTUtils::conjugate(input_output);
            input_output = input_output * Complex(1.0/n, 0);
        }         
    };

}


#endif //OMPFFT_HPP
