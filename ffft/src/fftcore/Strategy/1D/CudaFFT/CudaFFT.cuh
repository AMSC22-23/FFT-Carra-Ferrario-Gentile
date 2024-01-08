#ifndef CUDAFFT_HPP
#define CUDAFFT_HPP

#include <iostream>
#include <cuda/std/complex>

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

template <typename FloatingType>
using ComplexCuda = cuda::std::complex<FloatingType>;

/**
 * @brief This kernel performs one stage of the Cooley-Tukey FFT algorithm. Each thread performs a single butterfly operation, so the number of threads is n/2 at each stage.
 * @todo Optimizations: precompute roots of unity, use shared memory, try stockham (no bitreversal)
*/
template <typename FloatingType>
__global__ void d_butterfly_kernel(ComplexCuda<FloatingType> *input_output, int n, int m2, ComplexCuda<FloatingType> wm)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int k, j;
    ComplexCuda<FloatingType> w, t, u;

    if (tid < n / 2)
    {
        j = tid % m2;
        k = (tid / m2) * m2 * 2;

        w = pow(wm, j);
        t = w * input_output[k + j + m2];
        u = input_output[k + j];

        input_output[k + j] = u + t;
        input_output[k + j + m2] = u - t;
    }
}

/**
 * @brief This helper function is called by the bit reversal permutation kernel. It reverses the bits of an unsigned integer.
*/
__device__ unsigned int d_reverse_bits(unsigned int n, int log2n)
{
    unsigned int result = 0;
    for (int i = 0; i < log2n; i++)
    {
        if (n & (1 << i))
        {
            result |= 1 << (log2n - 1 - i);
        }
    }
    return result;
}

/**
 * @brief This kernel performs the bit reversal permutation on the input array. It is called before the Cooley-Tukey FFT algorithm. Each thread performs a single swap, so the number of threads is n.
*/
template <typename FloatingType>
__global__ void d_bit_reversal_permutation(ComplexCuda<FloatingType> *input_output, int n, int log2n)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    ComplexCuda<FloatingType> swap;

    if (tid < n)
    {
        unsigned int rev = d_reverse_bits(tid, log2n);
        if (rev > tid)
        {
            swap = input_output[tid];
            input_output[tid] = input_output[rev];
            input_output[rev] = swap;
        }
    }
}

namespace fftcore
{

    /**
     * @brief This class implements the FFT algorithm on the GPU using CUDA. For now it uses the Cooley-Tukey algorithm, but it will be extended to use the Stockham algorithm. 
     * @todo Make it compatible with cmake, add Stockham algorithm, add support for real input
     * @author Lorenzo Gentile
     * @date 2023-12-19
    */
    template <typename FloatingType = double>
    class CudaFFT : public FFT_1D<FloatingType>
    {
    public:
        using typename FFT_1D<FloatingType>::RTensor_1D;
        using typename FFT_1D<FloatingType>::CTensor_1D;

        using Complex = std::complex<FloatingType>;
        using ComplexCuda = ComplexCuda<FloatingType>;

        //The constructor calls cudaFree(0) to initialize CUDA memory context. This is dne to avoid wrong timing due to lazy initialization.
        CudaFFT(){
            cudaFree(0);
        }

        void fft(const CTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(CTensor_1D &, FFTDirection) const;

        ~CudaFFT() = default;
    };

    template <typename FloatingType>
    void CudaFFT<FloatingType>::fft(const CTensor_1D &input, CTensor_1D &output, FFTDirection fftDirection) const
    {
        memcpy(output.data(), input.data(), input.size() * sizeof(ComplexCuda));
        fft(output, fftDirection);
    };

    template <typename FloatingType>
    void CudaFFT<FloatingType>::fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const
    {
        throw NotSupportedException("Operation is not supported");
    };

    template <typename FloatingType>
    void CudaFFT<FloatingType>::fft(CTensor_1D &input_output, fftcore::FFTDirection fftDirection) const
    {

        int n = input_output.size();

        int threadsPerBlock = 128;
        int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
        int log2n = std::log2(n);

        //conjugate if inverse
        if(fftDirection == FFT_INVERSE){
            FFTUtils::conjugate(input_output);
        }

        // allocate memory on device
        ComplexCuda *d_input_output;
        gpuErrchk(cudaMalloc((void **)&d_input_output, n * sizeof(ComplexCuda)));

        // copy input to device
        gpuErrchk(cudaMemcpy(d_input_output, input_output.data(), n * sizeof(ComplexCuda), cudaMemcpyHostToDevice));

        // bit reversal permutation on device
        d_bit_reversal_permutation<<<numBlocks, threadsPerBlock>>>(d_input_output, n, log2n);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        numBlocks = (n / 2 + threadsPerBlock - 1) / threadsPerBlock;
        // Cooley-Tukey iterative FFT
        int m, m2;
        ComplexCuda wm;
        for (int s = 1; s <= log2n; ++s)
        {

            m = 1 << s;                              // 2^s
            m2 = m >> 1;                             // m2 = m/2
            wm = exp(ComplexCuda(0, -2 * M_PI / m)); // w_m = e^(-2*pi/m)
            d_butterfly_kernel<<<numBlocks, threadsPerBlock>>>(d_input_output, n, m2, wm);

            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }

        // copy output to host
        gpuErrchk(cudaMemcpy(input_output.data(), d_input_output, n * sizeof(ComplexCuda), cudaMemcpyDeviceToHost));

        // free memory on device
        gpuErrchk(cudaFree(d_input_output));

        //re-conjugate and scale if inverse
        if(fftDirection == FFT_INVERSE){
            FFTUtils::conjugate(input_output);
            input_output = input_output * Complex(1.0/n, 0);
        }

    };

}

#endif // CUDAFFT_HPP