#ifndef CUFFTFFT_CUH
#define CUFFTFFT_CUH

#include "CudaUtils.cuh"
#include <iostream>
#include <complex>
#include <cufft.h>

namespace fftcore
{

    /**
     * @brief This class implements cuFFT and is used for benchmarking purposes.
     * @author Lorenzo Gentile
     * @date 2024-01-02
    */
    template <typename FloatingType = double>
    class cufftFFT : public FFT_1D<FloatingType>
    {
    public:
        using typename FFT_1D<FloatingType>::RTensor_1D;
        using typename FFT_1D<FloatingType>::CTensor_1D;

        //The constructor calls cudaFree(0) to initialize CUDA memory context. This is done to avoid wrong timing due to lazy initialization.
        cufftFFT(){
            cudaFree(0);
        }

        void fft(const CTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(CTensor_1D &, FFTDirection) const;

        ~cufftFFT() = default;
    };

    template <typename FloatingType>
    void cufftFFT<FloatingType>::fft(const CTensor_1D &input, CTensor_1D &output, FFTDirection fftDirection) const
    {
        output = input; //deep copy
        fft(output, fftDirection);
    };

    template <typename FloatingType>
    void cufftFFT<FloatingType>::fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const
    {
        throw NotSupportedException("Operation is not supported");
    };

    template <typename FloatingType>
    void cufftFFT<FloatingType>::fft(CTensor_1D &input_output, fftcore::FFTDirection fftDirection) const
    {

        int n = input_output.size();
        
        cufftHandle plan;

        cufftDoubleComplex *d_data;
        cudaMalloc((void **)&d_data, n * sizeof(cufftDoubleComplex));
        cudaMemcpy(d_data, input_output.data(), n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

        cufftPlan1d(&plan, n, CUFFT_Z2Z, 1);
        cufftExecZ2Z(plan, d_data, d_data, fftDirection == FFT_FORWARD ? CUFFT_FORWARD : CUFFT_INVERSE);
        cudaDeviceSynchronize();

        if(fftDirection == FFT_INVERSE){
            d_scale<<<(n + 255) / 256, 256>>>(reinterpret_cast<ComplexCuda<FloatingType>*>(d_data), n);
        }

        cudaMemcpy(input_output.data(), d_data, n * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
        
        cufftDestroy(plan);
        cudaFree(d_data);
    };


}

#endif // CUFFTFFT_CUH