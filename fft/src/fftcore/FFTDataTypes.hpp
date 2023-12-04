#ifndef H_FFTDATATYPES_H
#define H_FFTDATATYPES_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace fftcore{

	//1D
	template<typename FloatingType>
	using CTensor_1D = Eigen::Tensor<std::complex<FloatingType>, 1>;
	template<typename FloatingType>
	using RTensor_1D = Eigen::Tensor<FloatingType, 1>;

	//2D
	template<typename FloatingType>
	using CTensor_2D = Eigen::Tensor<std::complex<FloatingType>, 2>;
	template<typename FloatingType>
	using RTensor_2D = Eigen::Tensor<FloatingType, 2>;

	//3D
	template<typename FloatingType>
	using CTensor_3D = Eigen::Tensor<std::complex<FloatingType>, 3>;
	template<typename FloatingType>
	using RTensor_3D = Eigen::Tensor<FloatingType, 3>;

    //TensorBase generic type aliases 
    template<int Rank, typename FloatingType = double>
    using CTensorBase = TensorFFTBase<std::complex<FloatingType>, Rank>;
    template<int Rank, typename FloatingType = double>
	using RTensorBase = TensorFFTBase<FloatingType, Rank>;

	// FFT_DIRECTION
	enum FFTDirection{FFT_FORWARD,FFT_INVERSE};
}

#endif
