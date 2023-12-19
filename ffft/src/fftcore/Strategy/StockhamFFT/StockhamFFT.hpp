#ifndef STOCKHAMFFT_HPP
#define STOCKHAMFFT_HPP

#include "../../FFTSolver.hpp"
#include "../../utils/FFTUtils.hpp"
#include <iostream>

using namespace std;
namespace fftcore{

    /**
     * @brief This class provides a sequential FFT implementation based on the Stockham algorithm.
     * The Stockham algorithm is a variation of the traditional Cooley-Tukey algorithm, with the advantage of avoiding the bit reversal step. Only 1D FFTs are supported for now.
     * @todo Implement R2C transform
     * @author Lorenzo Gentile
     * @date 2023-12-18
    */

    template<typename FloatingType = double>
    class StockhamFFT : public FFT_1D<FloatingType>
    {
    public:
        using typename FFT_1D<FloatingType>::RTensor_1D;
        using typename FFT_1D<FloatingType>::CTensor_1D;
        
        void fft(const CTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(CTensor_1D &, FFTDirection) const;

        ~StockhamFFT() = default;
    };

    template <typename FloatingType>
    void StockhamFFT<FloatingType>::fft(const CTensor_1D &input, CTensor_1D &output, FFTDirection FFTDirection) const
    {
        output = input; //deep copy
        fft(output, FFTDirection);
    };

    template <typename FloatingType>
    void StockhamFFT<FloatingType>::fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const
    {
        throw NotSupportedException("Operation is not supported");
    };

    template <typename FloatingType>
    void StockhamFFT<FloatingType>::fft(CTensor_1D &input_output, FFTDirection fftDirection) const
    {
        using Complex = std::complex<FloatingType>;

        CTensor_1D &input = input_output;
        CTensor_1D buffer(input.size());
        
        int n = input.size();
        int n2 = n >> 1; // n2 = n/2
        assert(!(n & (n - 1)) && "FFT length must be a power of 2.");

        int log2n = std::log2(n);

        //conjugate if inverse
        if(fftDirection == fftcore::FFT_INVERSE){
            FFTUtils::conjugate(input);
        }

        Complex w, wm, t, u;
        int m, m2;
        // Stockham iterative FFT
        for(int s = 1; s <= log2n; ++s){

            m = 1 << s;  // 2^s
            m2 = m >> 1; // m2 = m/2 
            wm = exp(Complex(0, -2 * M_PI / m)); // w_m = e^(-2*pi/m)

            for(int k = 0; k < n; k += m){
                w = Complex(1, 0);
                for(int j = 0; j < m2; ++j){
                    u = input[k / 2 + j];
                    t = w * input[n2 + k / 2 + j];

                    buffer[k + j] = u + t;
                    buffer[k + j + m2] = u - t;

                    w *= wm;
                }
            }
            std::swap(input, buffer);
        }

        //re-conjugate and scale if inverse
        if(fftDirection == fftcore::FFT_INVERSE){
            FFTUtils::conjugate(input);
            input = input * Complex(1.0/n, 0);
        }
    };

}

#endif //STOCKHAMFFT_HPP