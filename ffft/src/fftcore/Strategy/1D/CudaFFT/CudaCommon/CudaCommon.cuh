#ifndef CUDAUTILS_CUH
#define CUDAUTILS_CUH

#define EIGEN_NO_CUDA
#include <cuda/std/complex>
#include "../../../../utils/FFTDataTypes.hpp"
#include "../../../../Strategy/FFTStrategy.hpp"

namespace fftcore
{
    namespace cudakernels
    {
        template <typename FloatingType>
        using ComplexCuda = cuda::std::complex<FloatingType>;
        using fftcore::TensorIdx;

        //constant memory variables are declared here because they are common to all implementations and they would collide if declared in the implementation files
        __constant__ char d_fft_sign; // -1 for forward, 1 for inverse
        __constant__ TensorIdx d_n2; // n2 = n / 2 stored in constant memory to avoid passing it as a parameter to the kernel

        /**
         * @brief Bit reversal permutation on GPU
         * @details This function uses the CUDA intrinsic __brev to reverse the bits of the indices, which maps to 
         * a single instruction on the GPU.
        */
        template <typename FloatingType>
        __global__ void d_bit_reversal_permutation(ComplexCuda<FloatingType> *input_output, int n, int log2n);

        /**
         * @brief Helper kernel which scales a complex vector by 1/n. It is used in the inverse FFT.
        */
        template <typename FloatingType>
        __global__ void d_scale(ComplexCuda<FloatingType> *input_output, int n);

    } // namespace cudakernels

    namespace cudautils
    {

        #define gpuErrchk(ans)                        \
                {                                         \
                    gpuAssert((ans), __FILE__, __LINE__); \
                }

        //function to print cuda errors
        void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);

        //function to print gpu info 
        void printGPUInfo();
    } // namespace cudautils

} // namespace fftcore

#endif // CUDAUTILS_CUH