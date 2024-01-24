#ifndef FFTCORE_HPP
#define FFTCORE_HPP

/*! \mainpage AMSC project powered by Carrà Edoardo, Gentile Lorenzo, Ferrario Daniele.
 *
 * \section Introduction
 * 
 * FFTCore is a parallel C++ library designed for computing the Fast Fourier Transform design to be highly extensible, and easy to integrate in different applications.
 * The Library contains different sequential and parallel implementation to compute the FFT in one or higher dimension based on Stockham and Cooley-Tuckey algorithms. 
 *  
 * \subsection Library folder structure
 *   - `fftcore/`: this directory contains the core files of the library.
 *   - `FFTSolver.hpp`: this header file provides direct access for users to perform FFT operations.
 *   - `Strategy/`: this directory contains different strategies for performing the FFT. Some basic implementations are provided by default as the sequential one, MPI, OMP and CUDA.
 *   - `Tensor/`: this directory contains the main tools to manipulate Eigen's tensor and do pre-processing and post-processing.
 * \subsection  Supported strategies
 *  At the moment the supported strategies are:
 *  - Stockham only 1 dimensioanl.
 *  - Cooley-Tuckey 1,2 and 3 dimensional.
 *  - Cooley-Tuckey MPI 1,2 and 3 dimensional.
 *  - Cooley-Tuckey OMP 1,2 and 3 dimensional.
 *  - Cooley-Tuckey CUDA 1,2 and 3 dimensional.
 */

/**
* Public headers of the library. Include this file in your project to use the ffft library.
*
* @author Edoardo Carrà
*/

#ifdef __CUDACC__
#define EIGEN_NO_CUDA //to avoid compilation warnings, eigen is only used in host code
#endif

#include "../../src/fftcore/FFTSolver.hpp"

// strategies 1D
#include "../../src/fftcore/Strategy/1D/SequentialFFT/SequentialFFT.hpp"
#include "../../src/fftcore/Strategy/fftwFFT/fftwFFT.hpp"
#include "../../src/fftcore/Strategy/1D/MPIFFT/MPIFFT.hpp"
#include "../../src/fftcore/Strategy/1D/OpenMP/OmpFFT.hpp"
#include "../../src/fftcore/Strategy/1D/StockhamFFT/StockhamFFT.hpp"

#ifdef USE_CUDA
#include "../../src/fftcore/Strategy/1D/CudaFFT/CudaStockhamFFT/CudaStockhamFFT.cuh"
#include "../../src/fftcore/Strategy/1D/CudaFFT/CudaCooleyTukeyFFT/CudaCooleyTukeyFFT.cuh"
#endif

// strategies 2D
#include "../../src/fftcore/Strategy/2D/SequentialFFT_2D/SequentialFFT_2D.hpp"
#include "../../src/fftcore/Strategy/2D/OpenMP_2D/OmpFFT_2D.hpp"
#include "../../src/fftcore/Strategy/2D/MPIFFT_2D/MPIFFT_2D.hpp"
#include "../../src/fftcore/Strategy/2D/MPI_OMP_2D/MPI_OMP_FFT_2D.hpp"


// strategies 3D
#include "../../src/fftcore/Strategy/3D/SequentialFFT_3D/SequentialFFT_3D.hpp"
#include "../../src/fftcore/Strategy/3D/OpenMP_3D/OmpFFT_3D.hpp"
#include "../../src/fftcore/Strategy/3D/MPIFFT_3D/MPIFFT_3D.hpp"


//Timer
#include "../../src/fftcore/Timer/Timer.hpp"

// Tensor
#include "../../src/fftcore/Tensor/TensorFFTBase.hpp"

// Utilities
#include "../../src/fftcore/utils/FFTUtils.hpp"
//#include "../../src/fftcore/utils/MtxFilesIO.hpp"
//#include "../../src/fftcore/utils/MtxFilesIOUtils.hpp"


#endif //FFTCORE_HPP