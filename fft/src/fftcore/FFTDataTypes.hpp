#ifndef H_FFTDATATYPES_H
#define H_FFTDATATYPES_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace fftcore{

	//1D
	template<typename D>
	using CTensor_1D = Eigen::Tensor<std::complex<D>, 1>;
	template<typename D>
	using RTensor_1D = Eigen::Tensor<D, 1>;

	//2D
	template<typename D>
	using CTensor_2D = Eigen::Tensor<std::complex<D>, 2>;
	template<typename D>
	using RTensor_2D = Eigen::Tensor<D, 2>;

	//3D
	template<typename D>
	using CTensor_3D = Eigen::Tensor<std::complex<D>, 3>;
	template<typename D>
	using RTensor_3D = Eigen::Tensor<D, 3>;

	// FFT_DIRECTION
	enum FFTDirection{FFT_FORWARD,FFT_INVERSE};
}

#endif
