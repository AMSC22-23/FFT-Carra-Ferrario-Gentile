#ifndef FFTSTRATEGY_HPP
#define FFTSTRATEGY_HPP

#include "../utils/FFTDataTypes.hpp"

namespace fftcore{
	/**
	 * The solving Strategy which define the computation 
	 * of fft algorithms.
	 * @TODO: better description
	*/
	template<int Rank, typename FloatingType>
	class FFTStrategy{

		using CTensor = Eigen::Tensor<std::complex<FloatingType>, Rank>;
		using RTensor = Eigen::Tensor<FloatingType, Rank>;

		public:

			/**
			* Complex to Complex out-of-place FFT
			*/ 
			virtual void fft(const CTensor&, CTensor&, FFTDirection) const = 0;
			
			/**
			* Real to Complex out-of-place FFT
			*/ 
			virtual void fft(const RTensor&, CTensor&, FFTDirection) const = 0;

			/**
			* In-place FFT
			*/ 
			virtual void fft(CTensor&, FFTDirection) const = 0;
			
			virtual ~FFTStrategy() = default;
	};

	/**
	 * UTILITY STRATEGY DIMENSION DEFINERS MARKER CLASSES
	 * @TODO: useful for code clarity?
	*/

	/**
	 * Strategies which extend this marker class refer
	 * to 1D vectors algorithms.
	*/
	template<typename FloatingType>
	class FFT_1D :
	public FFTStrategy<1, FloatingType>{
		public:
			using RTensor_1D = Eigen::Tensor<FloatingType, 1>;
			using CTensor_1D = Eigen::Tensor<std::complex<FloatingType>, 1>;

			virtual ~FFT_1D() = default;
	};

	/**
	 * Strategies which extend this marker class refer
	 * to 2D matrices algorithms.
	*/
	template<typename FloatingType>
	class FFT_2D :
	public FFTStrategy<2, FloatingType>{

		public:
			using RTensor_2D = Eigen::Tensor<FloatingType, 2>;
			using CTensor_2D = Eigen::Tensor<std::complex<FloatingType>, 2>;
			// @TODO: Maybe change this
			using RTensor_1D = Eigen::Tensor<FloatingType, 1>;
			using CTensor_1D = Eigen::Tensor<std::complex<FloatingType>, 1>;
			virtual ~FFT_2D() = default;
	};

	/**
	 * Strategies which extend this marker class refer
	 * to 3D Tensors algorithms.
	*/
	template<typename FloatingType>
	class FFT_3D :
	public FFTStrategy<3, FloatingType>{

		public:
			using RTensor_3D = Eigen::Tensor<FloatingType, 3>;
			using CTensor_3D = Eigen::Tensor<std::complex<FloatingType>, 3>;
			
			virtual ~FFT_3D() = default;
	};

}

#endif // FFTSTRATEGY_HPP