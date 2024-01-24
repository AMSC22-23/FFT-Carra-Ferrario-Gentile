#ifndef CUDACOOLEYTUKEY_HPP
#define CUDACOOLEYTUKEY_HPP

#ifdef __CUDACC__
#include "../CudaCommon/CudaCommon.cuh"

namespace fftcore::cudakernels
{
    /**
     * @brief This kernel computes one stage of Cooley-Tukey FFT algorithm. It is called log2(n) times by the cpu driver code.
    */
    template <typename FloatingType>
    __global__ void d_butterfly_kernel_cooleytukey(ComplexCuda<FloatingType> * __restrict__ input_output, unsigned m2);
}
#endif // __CUDACC__

namespace fftcore
{   
    /**
     * /**
     * @brief CUDA implementation of the 1 dimensional FFT using Cooley-Tukey algorithm
     * @details The algorithm is a radix-2 decimation-in-time (DIT) FFT. It needs a bit reversal permutation of the input data, and when called in inverse mode it computes the roots of unity with opposite sign.
     * @todo Implement precomputation of twiddle factors in constant memory.
     * @author Lorenzo Gentile
     * @date 2024-01-09
    */
    template <typename FloatingType = double>
    class CudaCooleyTukeyFFT : public FFT_1D<FloatingType>
    {
    public:
        using typename FFT_1D<FloatingType>::RTensor_1D;
        using typename FFT_1D<FloatingType>::CTensor_1D;

        CudaCooleyTukeyFFT();

        void fft(const CTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(CTensor_1D &, FFTDirection) const;

        ~CudaCooleyTukeyFFT() = default;
    private:
        static constexpr int THREADS_PER_BLOCK = 32;
    };

} // namespace fftcore

#endif // CUDACOOLEYTUKEY_HPP