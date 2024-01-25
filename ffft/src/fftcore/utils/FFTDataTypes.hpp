#ifndef FFTDATATYPES_HPP
#define FFTDATATYPES_HPP

#include <stdexcept>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../Tensor/TensorFFTBase.hpp"


namespace fftcore{

    /**
     * @brief Public typename for TensotFFTBase templated on complex scalar type.
    */
    template<int Rank, typename FloatingType = double>
    using CTensorBase = TensorFFTBase<std::complex<FloatingType>, Rank>;

    /**
     * @brief Public typename for TensotFFTBase templated on real scalar type.
    */
    template<int Rank, typename FloatingType = double>
	using RTensorBase = TensorFFTBase<FloatingType, Rank>;

	/**
     * @brief Represents the direction of the fft.  
     */
	enum FFTDirection{
                      FFT_FORWARD, /** Forward FFT direction*/
                      FFT_INVERSE  /** Inverse FFT direction */
                      };

    /**
     * @brief A runtime exception for methods that are not supported yet
     */
    class NotSupportedException : public std::runtime_error {
        public:
        NotSupportedException(const std::string& message): std::runtime_error(message) {}
    };

    /**
     * Type for tensor sizes and indices
     */
    using TensorIdx = Eigen::Index;
}

#endif //FFTDATATYPES_HPP
