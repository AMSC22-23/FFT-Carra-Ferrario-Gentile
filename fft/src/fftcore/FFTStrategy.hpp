#ifndef H_FFTSTRATEGY_H
#define H_FFTSTRATEGY_H

#include "FFTDataTypes.hpp"

namespace fftcore{
	/**
	 * The solving Strategy which define the computation 
	 * of fft algorithms.
	 * @TODO: better description
	*/
	template<typename DataType, int Rank>
	class FFTStrategy{

		using CTensor = Eigen::Tensor<std::complex<DataType>, Rank>;

		using RTensor = Eigen::Tensor<DataType, Rank>;

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
	template<typename DataType>
	class FFT_1D :
	public FFTStrategy<DataType, 1>{

		public:
			
			virtual ~FFT_1D() = default;
	};

	/**
	 * Strategies which extend this marker class refer
	 * to 2D matrices algorithms.
	*/
	template<typename DataType>
	class FFT_2D :
	public FFTStrategy<DataType, 2>{

		public:
		
			
			virtual ~FFT_2D() = default;
	};

	/**
	 * Strategies which extend this marker class refer
	 * to 3D Tensors algorithms.
	*/
	template<typename DataType>
	class FFT_3D :
	public FFTStrategy<DataType, 3>{

		public:

			virtual ~FFT_3D() = default;
	};

}

#endif