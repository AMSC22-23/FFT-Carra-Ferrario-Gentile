#ifndef OMPFFT_HPP
#define OMPFFT_HPP

#include <iostream>
#include <omp.h>
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
    void OmpFFT<FloatingType>::fft(CTensor_1D& input_output, fftcore::FFTDirection fft_direction) const {

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

        Complex w, wm, wm_i, t, u;
        int m, m2;
        // Cooley-Tukey iterative FFT
        for (int s = 1; s <= log2n; ++s) {
            m = 1 << s;         // 2 power s
            m2 = m >> 1;        // m2 = m/2 -1
            wm = exp(Complex(0, -2 * M_PI / m)); // w_m = e^(-2*pi/m)
            wm_i = exp(Complex(0, 2 * M_PI / m)); // w_m = e^(2*pi/m)

            for (int k = 0; k < n; k += m) {
			//	#pragma omp parallel for shared(input_output)
				
				w = Complex(1, 0);
				for (int j = 0; j < m2; ++j) {
					//if(fft_direction==FFT_FORWARD){
					//	w = wm*exp(Complex(0,(k*m2)+j));
					//}else{
					//	w = wm_i*exp(Complex(0,(k*m2)+j));
					//}
                    t = w * input_output[j + k + m2];
                    u = input_output[j + k];
                    
                    input_output[k] = u + t;
                    input_output[k + m2] = u - t;

					if(fft_direction==FFT_FORWARD)w = w*wm;
					else w=w*wm_i;
                }
            }
        }
    
        if(fft_direction == FFT_INVERSE){
            for(int i=0; i<n; i++){
                input_output[i] /= n;
            }
            
            // // Re-oredering
            // // @TODO: I don't know if it's correct, but it works (Ferra)
            // // Also, no need to conjugate anything apparently
            // for (unsigned int i = 1; i < n/2; ++i) {
            //     std::swap(input_output[i], input_output[n-i]);
            // }
            

        }
            
    };

}


#endif //OMPFFT_HPP