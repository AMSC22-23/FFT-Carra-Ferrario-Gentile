#ifndef CUDASTOCKHAMFFT_CUH
#define CUDASTOCKHAMFFT_CUH

#ifdef __CUDACC__
#include "../CudaCommon/CudaCommon.cuh"

namespace fftcore::cudakernels
{
    template <typename FloatingType>
    __global__ void d_butterfly_kernel_stockham(ComplexCuda<FloatingType> * __restrict__ d_input, ComplexCuda<FloatingType> * __restrict__ d_buffer, unsigned m2);
}
#endif // __CUDACC__

namespace fftcore
{

    /**
     * @brief CUDA implementation of the 1 dimensional FFT using Stockham algorithm
     * @details The algorithm is a radix-2 decimation-in-time (DIT) FFT. It avoids the bit reversal step, which is inherently cache unfriendly, at the cost of a more complex indexing scheme and the need for a temporary buffer. It achieves a better performance than the Cooley-Tukey algorithm on the GPU for large input sizes.
     * @todo Implement precomputation of twiddle factors in constant memory.
     * @author Lorenzo Gentile
     * @date 2024-01-09
    */
    template <typename FloatingType = double>
    class CudaStockhamFFT : public FFT_1D<FloatingType>
    {
    public:
        using typename FFT_1D<FloatingType>::RTensor_1D;
        using typename FFT_1D<FloatingType>::CTensor_1D;

        CudaStockhamFFT();

        void fft(const CTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(CTensor_1D &, FFTDirection) const;

        ~CudaStockhamFFT() = default;
    private:
        static constexpr unsigned int THREADS_PER_BLOCK = 32;
    };

} // namespace fftcore

#endif //CUDASTOCKHAMFFT_CUH