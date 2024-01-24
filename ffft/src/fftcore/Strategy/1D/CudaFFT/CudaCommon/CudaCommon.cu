#include "CudaCommon.cuh"

namespace fftcore
{
    namespace cudakernels
    {
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

        template <typename FloatingType>
        __global__ void d_scale(ComplexCuda<FloatingType> *input_output, int n){
            unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid < n)
            {
                input_output[tid] /= n;
            }
        }
    } // namespace cudakernels

    namespace cudautils
    {
        void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
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
    } // namespace cudautils

} // namespace fftcore

namespace fftcore::cudakernels
{
    //Explicit template instantiation
    template __global__ void cudakernels::d_bit_reversal_permutation<float>(ComplexCuda<float> *input_output, int n, int log2n);
    template __global__ void cudakernels::d_bit_reversal_permutation<double>(ComplexCuda<double> *input_output, int n, int log2n);

    template __global__ void cudakernels::d_scale<float>(ComplexCuda<float> *input_output, int n);
    template __global__ void cudakernels::d_scale<double>(ComplexCuda<double> *input_output, int n);
}

