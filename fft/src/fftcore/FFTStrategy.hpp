#ifndef H_FFTSTRATEGY_H
#define H_FFTSTRATEGY_H

#include "FFTDataTypes.hpp"

namespace fftcore{

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

	template<typename DataType>
	class FFT_1D :
	public FFTStrategy<DataType, 1>{

		public:
			/**
			* Complex to Complex out-of-place 1D-FFT
			*/ 
			virtual void fft(const CTensor_1D<DataType>&, CTensor_1D<DataType>&, FFTDirection) const = 0;
			
			/**
			* Real to Complex out-of-place 1D-FFT
			*/ 
			virtual void fft(const RTensor_1D<DataType>&, CTensor_1D<DataType>&, FFTDirection) const = 0;

			/**
			* In-place 1D-FFT
			*/ 
			virtual void fft(CTensor_1D<DataType>&, FFTDirection) const = 0;
			
			virtual ~FFT_1D() = default;
	};

	template<typename DataType>
	class FFT_2D :
	public FFTStrategy<DataType, 2>{

		public:
		
			/**
			* Complex to Complex out-of-place 2-D FFT
			*/ 
			virtual void fft(const CTensor_2D<DataType>&, CTensor_2D<DataType>&, FFTDirection) const = 0;
			
			/**
			* Real to Complex out-of-place 2-D FFT
			*/ 
			virtual void fft(const RTensor_2D<DataType>&, CTensor_2D<DataType>&, FFTDirection) const = 0;

			/**
			* In-place 2-D FFT
			*/ 
			virtual void fft(CTensor_2D<DataType>&, FFTDirection) const = 0;
			
			virtual ~FFT_2D() = default;
	};

	template<typename DataType>
	class FFT_3D :
	public FFTStrategy<DataType, 3>{

		public:

			/**
			* Complex to Complex out-of-place 3-D FFT
			*/ 
			virtual void fft(const CTensor_3D<DataType>&, CTensor_3D<DataType>&, FFTDirection) const = 0;
			
			/**
			* Real to Complex out-of-place 3-D FFT
			*/ 
			virtual void fft(const RTensor_3D<DataType>&, CTensor_3D<DataType>&, FFTDirection) const = 0;

			/**
			* In-place 3-D FFT
			*/ 
			virtual void fft(CTensor_3D<DataType>&, FFTDirection) const = 0;
			
			virtual ~FFT_3D() = default;
	};

}

#endif