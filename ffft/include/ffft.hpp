#ifndef FFTCORE_HPP
#define FFTCORE_HPP

/**
* Public headers of the library. Include this file in your project to use the ffft library
* @author Edoardo Carrà
*/

#include "../src/fftcore/FFTSolver.hpp"

// strategies 1D
#include "../src/fftcore/Strategy/1D/SequentialFFT/SequentialFFT.hpp"
#include "../src/fftcore/Strategy/1D/fftwFFT/fftwFFT.hpp"
#include "../src/fftcore/Strategy/1D/MPIFFT/MPIFFT.hpp"
#include "../src/fftcore/Strategy/1D/OpenMP/OmpFFT.hpp"
#include "../src/fftcore/Strategy/1D/StockhamFFT/StockhamFFT.hpp"

// strategies 2D
#include "../src/fftcore/Strategy/2D/SequentialFFT_2D/SequentialFFT_2D.hpp"



//Timer

#include "../src/fftcore/Timer/Timer.hpp"

// Tensor
#include "../src/fftcore/Tensor/TensorFFTBase.hpp"

// Utilities
#include "../src/fftcore/utils/FFTUtils.hpp"


#endif //FFTCORE_HPP