#ifndef CUDASTOCKHAMFFT_CUH
#define CUDASTOCKHAMFFT_CUH

#include "CudaUtils.cuh"

template <typename FloatingType>
__global__ void d_butterfly_kernel_stockham(ComplexCuda<FloatingType> * __restrict__ d_input, ComplexCuda<FloatingType> * __restrict__ d_buffer, int n2, int m2)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n2)
    {
        unsigned int r = n2 / m2, r2 = r * 2;
        unsigned int k, j;
        ComplexCuda<FloatingType> w, t, u;

        /* old version, not coalesced
        j = tid % m2;
        k = (tid / m2);
        */

        j = tid >> (__ffs(r) - 1); // j = tid / r
        k = tid & (r - 1); // k = tid % r

        w = exp(ComplexCuda<FloatingType>(0, d_fft_sign * M_PI * j / m2));
        //w = ComplexCuda<FloatingType>(cos(- M_PI * j / m2), sin(- M_PI * j / m2));

        u = d_input[j * r2 + k];
        t = w * d_input[j * r2 + k + r];

        d_buffer[j * r + k] = u + t;
        d_buffer[j * r + n2 + k] = u - t; 

        //printf("Block %d, thread %d : (%d, %d) -> (%d, %d) \n", blockIdx.x, threadIdx.x, j * r2 + k, j * r2 + k + r, j * r + k, j * r + n2 + k);
    }
}

namespace fftcore
{

    /**
     * @brief This class implements a version of the Stockham FFT algorithm on the GPU.
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

        using Complex = std::complex<FloatingType>;
        using ComplexCuda = ComplexCuda<FloatingType>;

        CudaStockhamFFT(){
            gpuErrchk( cudaFree(0) );  // initialize CUDA context to avoid delay on first call
        }

        void fft(const CTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(CTensor_1D &, FFTDirection) const;

        ~CudaStockhamFFT() = default;
    private:
        static constexpr unsigned int THREADS_PER_BLOCK = 32;
    };

    template <typename FloatingType>
    void CudaStockhamFFT<FloatingType>::fft(const CTensor_1D &input, CTensor_1D &output, FFTDirection fftDirection) const
    {
        output = input; //deep copy
        fft(output, fftDirection);
    };

    template <typename FloatingType>
    void CudaStockhamFFT<FloatingType>::fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const
    {
        throw NotSupportedException("Operation is not supported");
    };

    template <typename FloatingType>
    void CudaStockhamFFT<FloatingType>::fft(CTensor_1D &input_output, fftcore::FFTDirection fftDirection) const
    {

        int n = input_output.size(), n2 = n / 2;
        assert(!(n & (n - 1)) && "FFT length must be a power of 2.");

        int threadsPerBlock = THREADS_PER_BLOCK;
        int numBlocks = (n2 + threadsPerBlock - 1) / threadsPerBlock;
        int log2n = std::log2(n);

        // allocate memory on device
        ComplexCuda *d_input, *d_buffer;
        gpuErrchk(cudaMalloc((void **)&d_input, n * sizeof(ComplexCuda)));
        gpuErrchk(cudaMalloc((void **)&d_buffer, n * sizeof(ComplexCuda)));

        // copy input to device
        gpuErrchk(cudaMemcpy(d_input, input_output.data(), n * sizeof(ComplexCuda), cudaMemcpyHostToDevice));

        // set fftDirection on device constant memory (to avoid passing it as a parameter to the kernel)
        char sign = (fftDirection == FFT_FORWARD) ? 1 : -1;
        gpuErrchk( cudaMemcpyToSymbol(d_fft_sign, &sign, sizeof(char)) );

        for(int s = 1; s <= log2n; ++s){
            
            int m = 1 << s; 
            int m2 = m >> 1;

            d_butterfly_kernel_stockham<<<numBlocks, threadsPerBlock>>>(d_input, d_buffer, n2, m2);

            std::swap(d_input, d_buffer);
        }

        // scale output if inverse fft
        if(fftDirection == FFT_INVERSE){
            numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
            d_scale<<<numBlocks, threadsPerBlock>>>(d_input, n);
        }

        // copy output to host
        gpuErrchk(cudaMemcpy(input_output.data(), d_input, n * sizeof(ComplexCuda), cudaMemcpyDeviceToHost));

        // free memory on device
        gpuErrchk(cudaFree(d_input));
        gpuErrchk(cudaFree(d_buffer));

    };

}

#endif //CUDASTOCKHAMFFT_CUH