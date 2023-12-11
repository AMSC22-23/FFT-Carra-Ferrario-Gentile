#ifndef FFTCORE_HPP
#define FFTCORE_HPP

/**
* Public headers of the library. Include this file in your project to use the library
* @author Edoardo Carrà
*/

#include "../src/fftcore/FFTSolver.hpp"

// strategies
#include "../src/fftcore/Strategy/SequentialFFT/SequentialFFT.hpp"
#include "../src/fftcore/Strategy/fftwFFT/fftwFFT.hpp"
#include "../src/fftcore/Strategy/MPIFFT/MPIFFT.hpp"
#include "../src/fftcore/Strategy/OpenMP/OmpFFT.hpp"

//Timer

#include "../src/fftcore/Timer/Timer.hpp"

// Tensor
#include "../src/fftcore/Tensor/TensorFFTBase.hpp"

// Utilities
#include "../src/fftcore/utils/FFTUtils.hpp"


#endif //FFTCORE_HPP