#ifndef CUDAUTILS_CUH
#define CUDAUTILS_CUH

#include <cstdio>
#include <cuda/std/complex>

//function to print cuda errors
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

//function to print gpu info 
void printGPUInfo()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Device count: %d\n", deviceCount);
    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("Device %d name: %s\n", i, deviceProp.name);
        printf("Device %d compute capability: %d.%d\n", i, deviceProp.major, deviceProp.minor);
        printf("Device %d clock rate: %d\n", i, deviceProp.clockRate);
        printf("Device %d global memory: %lu\n", i, deviceProp.totalGlobalMem);
        printf("Device %d shared memory per block: %lu\n", i, deviceProp.sharedMemPerBlock);
        printf("Device %d registers per block: %d\n", i, deviceProp.regsPerBlock);
        printf("Device %d warp size: %d\n", i, deviceProp.warpSize);
        printf("Device %d memory pitch: %lu\n", i, deviceProp.memPitch);
        printf("Device %d max threads per block: %d\n", i, deviceProp.maxThreadsPerBlock);
        printf("Device %d max threads dimensions: (%d, %d, %d)\n", i, deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Device %d max grid size: (%d, %d, %d)\n", i, deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Device %d total constant memory: %lu\n", i, deviceProp.totalConstMem);
    }
}

namespace cudakernels
{
    template <typename FloatingType>
    using ComplexCuda = cuda::std::complex<FloatingType>;
    
    //constant memory variables are declared here because they are common to all implementations and they would collide if declared in the implementation files
    __constant__ char d_fft_sign; // -1 for forward, 1 for inverse
    __constant__ TensorIdx d_n2; // n2 = n / 2 stored in constant memory to avoid passing it as a parameter to the kernel

    /**
     * @brief Bit reversal permutation on GPU
     * @details This function uses the CUDA intrinsic __brev to reverse the bits of the indices, which maps to 
     * a single instruction on the GPU.
    */
    template <typename FloatingType>
    __global__ void d_bit_reversal_permutation(ComplexCuda<FloatingType> *input_output, int n, int log2n)
    {
        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        ComplexCuda<FloatingType> swap;

        if (tid < n)
        {
            //unsigned int rev = d_reverse_bits(tid, log2n);
            unsigned int rev = __brev(tid) >> (32 - log2n);
            if (rev > tid)
            {
                swap = input_output[tid];
                input_output[tid] = input_output[rev];
                input_output[rev] = swap;
            }
        }
    }

    /**
     * @brief Helper kernel which scales a complex vector by 1/n. It is used in the inverse FFT.
    */
    template <typename FloatingType>
    __global__ void d_scale(ComplexCuda<FloatingType> *input_output, int n){
        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < n)
        {
            input_output[tid] /= n;
        }
    }

}

#endif // CUDAUTILS_CUH