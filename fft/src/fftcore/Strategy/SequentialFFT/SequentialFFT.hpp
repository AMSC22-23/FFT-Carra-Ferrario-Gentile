#ifndef SEQUENTIALFFT_HPP
#define SEQUENTIALFFT_HPP

#include "../../FFTSolver.hpp"
#include "../../utils/FFTUtils.hpp"
#include <iostream>

using namespace std;
namespace fftcore{

    template<typename FloatingType = double>
    class SequentialFFT : public FFT_1D<FloatingType>
    {
    public:
        using typename FFT_1D<FloatingType>::RTensor_1D;
        using typename FFT_1D<FloatingType>::CTensor_1D;
        
        void fft(const CTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(CTensor_1D &, FFTDirection) const;

        ~SequentialFFT() = default;
    };

    template <typename FloatingType>
    void SequentialFFT<FloatingType>::fft(const CTensor_1D &, CTensor_1D &, FFTDirection) const
    {
        throw NotSupportedException("Operation is not supported");
    };

    template <typename FloatingType>
    void SequentialFFT<FloatingType>::fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const
    {
        throw NotSupportedException("Operation is not supported");
    };

    /**
     * @author: Lorenzo Gentile, Daniele Ferrario
     */
    template <typename FloatingType>
    void SequentialFFT<FloatingType>::fft(CTensor_1D &input_output, fftcore::FFTDirection fftDirection) const
    {

        using Complex = std::complex<FloatingType>;
        int n = input_output.size();
        assert(!(n & (n - 1)) && "FFT length must be a power of 2.");

        int log2n = std::log2(n);

        //conjugate if inverse
        if(fftDirection == fftcore::FFT_INVERSE){
            FFTUtils::conjugate(input_output);
        }

        // Bit-reversal permutation
        FFTUtils::bit_reversal_permutation(input_output);

        Complex w, wm, t, u;
        int m, m2;
        // Cooley-Tukey iterative FFT
        for (int s = 1; s <= log2n; ++s)
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
            input_output = input_output * Complex(1.0/n, 0);
        }
    };

}

#endif //SEQUENTIALFFT_HPP