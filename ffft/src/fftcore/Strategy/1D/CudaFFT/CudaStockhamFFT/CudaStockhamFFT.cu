#include "CudaStockhamFFT.cuh"
#include "../CudaCommon/CudaCommon.cuh"

namespace fftcore::cudakernels
{
    template <typename FloatingType>
    __global__ void d_butterfly_kernel_stockham(ComplexCuda<FloatingType> * __restrict__ d_input, ComplexCuda<FloatingType> * __restrict__ d_buffer, unsigned m2)
    {   
        using ComplexCuda = ComplexCuda<FloatingType>;
        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if (tid < d_n2)
        {
            unsigned int r = d_n2 >> (__ffs(m2) - 1); // r = n2 / m2
            unsigned int r2 = r << 1; // r2 = 2 * r
            unsigned int k, j;
            ComplexCuda w, t, u;

            /* old version, not coalesced
            j = tid % m2;
            k = (tid / m2);
            */

            j = tid >> (__ffs(r) - 1); // j = tid / r
            k = tid & (r - 1); // k = tid % r

            //w = exp(ComplexCuda(0, d_fft_sign * M_PI * j / m2));
            w = ComplexCuda(__cosf(d_fft_sign * M_PI * j / m2), __sinf(d_fft_sign * M_PI * j / m2));
            u = d_input[j * r2 + k];
            t = w * d_input[j * r2 + k + r];

            d_buffer[j * r + k] = u + t;
            d_buffer[j * r + d_n2 + k] = u - t; 

        }
    }
}

namespace fftcore
{
    using cudautils::gpuAssert;

    template <typename FloatingType>
    CudaStockhamFFT<FloatingType>::CudaStockhamFFT()
    {
        gpuErrchk( cudaFree(0) ); // initialize cuda context
    }

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
        using cudakernels::d_butterfly_kernel_stockham;
        using cudakernels::d_scale;
        using cudakernels::d_fft_sign, cudakernels::d_n2;
        using ComplexCuda = cudakernels::ComplexCuda<FloatingType>;

        const TensorIdx n = input_output.size(), n2 = n / 2;
        assert(!(n & (n - 1)) && "FFT length must be a power of 2.");

        unsigned int threadsPerBlock = THREADS_PER_BLOCK;
        unsigned int numBlocks = (n2 + threadsPerBlock - 1) / threadsPerBlock;
        unsigned int log2n = std::log2(n);

        // allocate memory on device
        ComplexCuda *d_input, *d_buffer;
        gpuErrchk(cudaMalloc((void **)&d_input, n * sizeof(ComplexCuda)));
        gpuErrchk(cudaMalloc((void **)&d_buffer, n * sizeof(ComplexCuda)));

        // copy input to device
        gpuErrchk(cudaMemcpy(d_input, input_output.data(), n * sizeof(ComplexCuda), cudaMemcpyHostToDevice));

        // set fftDirection on device constant memory (to avoid passing it as a parameter to the kernel)
        char sign = (fftDirection == FFT_FORWARD) ? 1 : -1;
        gpuErrchk( cudaMemcpyToSymbol(d_fft_sign, &sign, sizeof(char)) );

        // set n2 on device constant memory 
        gpuErrchk( cudaMemcpyToSymbol(d_n2, &n2, sizeof(TensorIdx)) );

        TensorIdx m, m2;
        for(unsigned int s = 1; s <= log2n; ++s){
            
            m = 1 << s; 
            m2 = m >> 1;

            d_butterfly_kernel_stockham<<<numBlocks, threadsPerBlock>>>(d_input, d_buffer, m2);

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

} // namespace fftcore

namespace fftcore
{
    template class CudaStockhamFFT<float>;
    template class CudaStockhamFFT<double>;
} // namespace fftcore