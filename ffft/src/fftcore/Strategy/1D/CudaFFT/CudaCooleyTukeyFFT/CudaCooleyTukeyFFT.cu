#include "CudaCooleyTukeyFFT.cuh"
#include "../CudaCommon/CudaCommon.cuh"

namespace fftcore::cudakernels
{
    template <typename FloatingType>
    __global__ void d_butterfly_kernel_cooleytukey(ComplexCuda<FloatingType> * __restrict__ input_output, unsigned m2)
    {
        using ComplexCuda = ComplexCuda<FloatingType>;

        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int k, j;
        ComplexCuda w, t, u;

        if (tid < d_n2){

            j = tid & (m2 - 1);  // j = tid % m2
            k = (tid >> (__ffs(m2) - 1)) * m2 * 2;  // k = (tid / m2) * m

            // w = exp(ComplexCuda(0, d_fft_sign * M_PI * j / m2));
            w = ComplexCuda(__cosf(d_fft_sign * M_PI * j / m2), __sinf(d_fft_sign * M_PI * j / m2));
            t = w * input_output[k + j + m2];
            u = input_output[k + j];

            input_output[k + j] = u + t;
            input_output[k + j + m2] = u - t;
        }
    }
}

namespace fftcore
{   

    using cudautils::gpuAssert;

    template <typename FloatingType>
    CudaCooleyTukeyFFT<FloatingType>::CudaCooleyTukeyFFT()
    {
        gpuErrchk( cudaFree(0) ); //initialize CUDA context
    }

    template <typename FloatingType>
    void CudaCooleyTukeyFFT<FloatingType>::fft(const CTensor_1D &input, CTensor_1D &output, FFTDirection fftDirection) const
    {
        output = input; //deep copy
        fft(output, fftDirection);
    };

    template <typename FloatingType>
    void CudaCooleyTukeyFFT<FloatingType>::fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const
    {
        throw NotSupportedException("Operation is not supported");
    };

    template <typename FloatingType>
    void CudaCooleyTukeyFFT<FloatingType>::fft(CTensor_1D &input_output, fftcore::FFTDirection fftDirection) const
        {
        
        using cudakernels::d_butterfly_kernel_cooleytukey;
        using cudakernels::d_bit_reversal_permutation;
        using cudakernels::d_scale;
        using cudakernels::d_fft_sign, cudakernels::d_n2;
        using ComplexCuda = cudakernels::ComplexCuda<FloatingType>;

        const TensorIdx n = input_output.size(), n2 = n / 2, log2n = std::log2(n);

        //allocate memory on device
        ComplexCuda *d_input_output;
        gpuErrchk( cudaMalloc((void **)&d_input_output, n * sizeof(ComplexCuda)) );
            
        //copy input to device
        gpuErrchk( cudaMemcpy(d_input_output, input_output.data(), n * sizeof(ComplexCuda), cudaMemcpyHostToDevice) );

        //bit reversal permutation on device
        unsigned int threadsPerBlock = THREADS_PER_BLOCK;
        unsigned int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
        d_bit_reversal_permutation<<<numBlocks, threadsPerBlock>>>(d_input_output, n, log2n);

        //set fftDirection on device constant memory
        char sign = (fftDirection == FFT_FORWARD) ? -1 : 1;
        gpuErrchk( cudaMemcpyToSymbol(d_fft_sign, &sign, sizeof(char)) );

        //set n2 on device constant memory
        gpuErrchk( cudaMemcpyToSymbol(d_n2, &n2, sizeof(TensorIdx)) );

        //Cooley-Tukey iterative FFT
        numBlocks = (n2 + threadsPerBlock - 1) / threadsPerBlock; //set number of blocks so that each thread will process 2 elements
        TensorIdx m, m2;
        for(unsigned int s = 1; s <= log2n; ++s){

            m = 1 << s;  // 2^s
            m2 = m >> 1; // m2 = m/2
            d_butterfly_kernel_cooleytukey<<<numBlocks, threadsPerBlock>>>(d_input_output, m2);

        }

        //scale output if inverse FFT
        if(fftDirection == FFT_INVERSE){
            numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
            d_scale<<<numBlocks, threadsPerBlock>>>(d_input_output, n);
        }

        //copy output to host
        gpuErrchk( cudaMemcpy(input_output.data(), d_input_output, n * sizeof(ComplexCuda), cudaMemcpyDeviceToHost) );

        //free memory on device
        gpuErrchk( cudaFree(d_input_output) );
    };

} // namespace fftcore

namespace fftcore
{
    template class CudaCooleyTukeyFFT<float>;
    template class CudaCooleyTukeyFFT<double>;
} // namespace fftcore