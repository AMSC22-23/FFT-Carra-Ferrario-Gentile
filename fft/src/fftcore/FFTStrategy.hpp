#ifndef H_FFTSTRATEGY_H
#define H_FFTSTRATEGY_H

#include "FFTDataTypes.hpp"

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

			virtual ~FFT_3D() = default;
	};

}

#endif