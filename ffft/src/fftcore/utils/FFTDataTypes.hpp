#ifndef FFTDATATYPES_HPP
#define FFTDATATYPES_HPP

#include <stdexcept>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../Tensor/TensorFFTBase.hpp"


namespace fftcore{

    //TensorBase generic type aliases 
    template<int Rank, typename FloatingType = double>
    using CTensorBase = TensorFFTBase<std::complex<FloatingType>, Rank>;

    template<int Rank, typename FloatingType = double>
	using RTensorBase = TensorFFTBase<FloatingType, Rank>;

	// FFT_DIRECTION
	enum FFTDirection{FFT_FORWARD,FFT_INVERSE};

    class NotSupportedException : public std::runtime_error {
        public:
        NotSupportedException(const std::string& message): std::runtime_error(message) {}
    };
}

#endif //FFTDATATYPES_HPP
