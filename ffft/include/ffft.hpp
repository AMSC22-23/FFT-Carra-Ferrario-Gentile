#ifndef FFTCORE_HPP
#define FFTCORE_HPP

/**
* Public headers of the library. Include this file in your project to use the ffft library
* @author Edoardo Carrà
*/

#ifdef __CUDACC__
#define EIGEN_NO_CUDA //to avoid compilation warnings, eigen is only used in host code
#endif

#include "../src/fftcore/FFTSolver.hpp"

// strategies 1D
#include "../src/fftcore/Strategy/1D/SequentialFFT/SequentialFFT.hpp"
#include "../src/fftcore/Strategy/fftwFFT/fftwFFT.hpp"
#include "../src/fftcore/Strategy/1D/MPIFFT/MPIFFT.hpp"
#include "../src/fftcore/Strategy/1D/OpenMP/OmpFFT.hpp"
#include "../src/fftcore/Strategy/1D/StockhamFFT/StockhamFFT.hpp"

#ifdef __CUDACC__
#include "../src/fftcore/Strategy/1D/CudaFFT/CudaCooleyTukeyFFT.cuh"
#include "../src/fftcore/Strategy/1D/CudaFFT/CudaStockhamFFT.cuh"
#include "../src/fftcore/Strategy/1D/CudaFFT/cufftFFT.cuh"
#endif

// strategies 2D
#include "../src/fftcore/Strategy/2D/SequentialFFT_2D/SequentialFFT_2D.hpp"
#include "../src/fftcore/Strategy/2D/OpenMP_2D/OmpFFT_2D.hpp"
#include "../src/fftcore/Strategy/2D/MPIFFT/MPIFFT_2D.hpp"


// strategies 3D
#include "../src/fftcore/Strategy/3D/SequentialFFT_3D/SequentialFFT_3D.hpp"
#include "../src/fftcore/Strategy/3D/OpenMP_3D/OmpFFT_3D.hpp"
#include "../src/fftcore/Strategy/3D/MPIFFT_3D/MPIFFT_3D.hpp"


//Timer
#include "../src/fftcore/Timer/Timer.hpp"

// Tensor
#include "../src/fftcore/Tensor/TensorFFTBase.hpp"

// Utilities
#include "../src/fftcore/utils/FFTUtils.hpp"


#endif //FFTCORE_HPP