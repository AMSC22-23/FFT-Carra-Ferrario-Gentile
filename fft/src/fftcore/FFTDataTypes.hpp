#ifndef H_FFTDATATYPES_H
#define H_FFTDATATYPES_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace fftcore{

    //TensorBase generic type aliases 
    template<int Rank, typename FloatingType = double>
    using CTensorBase = TensorFFTBase<std::complex<FloatingType>, Rank>;
    template<int Rank, typename FloatingType = double>
	using RTensorBase = TensorFFTBase<FloatingType, Rank>;

	// FFT_DIRECTION
	enum FFTDirection{FFT_FORWARD,FFT_INVERSE};
}

#endif
