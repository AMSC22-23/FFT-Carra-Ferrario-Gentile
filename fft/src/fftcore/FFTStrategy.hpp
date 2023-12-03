#ifndef H_FFTSTRATEGY_H
#define H_FFTSTRATEGY_H

#include <unsupported/Eigen/CXX11/Tensor>
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
			using CTensor_1D = Eigen::Tensor<std::complex<DataType>, 1>;

			using RTensor_1D = Eigen::Tensor<DataType, 1>;
			/**
			* Complex to Complex out-of-place 1D-FFT
			*/ 
			virtual void fft(const CTensor_1D&, CTensor_1D&, FFTDirection) const = 0;
			
			/**
			* Real to Complex out-of-place 1D-FFT
			*/ 
			virtual void fft(const RTensor_1D&, CTensor_1D&, FFTDirection) const = 0;

			/**
			* In-place 1D-FFT
			*/ 
			virtual void fft(CTensor_1D&, FFTDirection) const = 0;
			
			virtual ~FFT_1D() = default;
	};

	template<typename DataType>
	class FFT_2D :
	public FFTStrategy<DataType, 2>{

		public:
			using CTensor_2D = Eigen::Tensor<std::complex<DataType>, 2>;

			using RTensor_2D = Eigen::Tensor<DataType, 2>;
			
			/**
			* Complex to Complex out-of-place 2-D FFT
			*/ 
			virtual void fft(const CTensor_2D&, CTensor_2D&, FFTDirection) const = 0;
			
			/**
			* Real to Complex out-of-place 2-D FFT
			*/ 
			virtual void fft(const RTensor_2D&, CTensor_2D&, FFTDirection) const = 0;

			/**
			* In-place 2-D FFT
			*/ 
			virtual void fft(CTensor_2D&, FFTDirection) const = 0;
			
			virtual ~FFT_2D() = default;
	};

	template<typename DataType>
	class FFT_3D :
	public FFTStrategy<DataType, 3>{

		public:
			using CTensor_3D = Eigen::Tensor<std::complex<DataType>, 3>;

			using RTensor_3D = Eigen::Tensor<DataType, 3>;

			/**
			* Complex to Complex out-of-place 3-D FFT
			*/ 
			virtual void fft(const CTensor_3D&, CTensor_3D&, FFTDirection) const = 0;
			
			/**
			* Real to Complex out-of-place 3-D FFT
			*/ 
			virtual void fft(const RTensor_3D&, CTensor_3D&, FFTDirection) const = 0;

			/**
			* In-place 3-D FFT
			*/ 
			virtual void fft(CTensor_3D&, FFTDirection) const = 0;
			
			virtual ~FFT_3D() = default;
	};

}

#endif