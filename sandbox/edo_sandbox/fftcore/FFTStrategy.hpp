#ifndef _FFTSTRATEGY_H
#define _FFTSTRATEGY_H

#include "FFTDataTypes.hpp"
#include <unsupported/Eigen/CXX11/Tensor>

namespace fftcore{
	
	template<typename DataType, int Rank>
	class FFTStrategy{
		public:
			FFTStrategy(){};
			
			//--------------------------------
			//---------------1D---------------
			//--------------------------------
			virtual void fft_1D_C2C(const Eigen::Tensor<DataType,1>&, Eigen::Tensor<DataType,1>, FFTDirection) const = 0;

			virtual void fft_1D_R2C(const Eigen::Tensor<DataType,1>&, Eigen::Tensor<DataType,1>, FFTDirection) const = 0;
			
			virtual void fft_1D_C2C(Eigen::Tensor<DataType,1>&, FFTDirection) const = 0;
			
			virtual void fft_1D_R2C(Eigen::Tensor<DataType,1>&, FFTDirection) const = 0;

			//--------------------------------
			//---------------2D---------------
			//--------------------------------			
			virtual void fft_2D_C2C(const Eigen::Tensor<DataType,2>&, Eigen::Tensor<DataType,1>, FFTDirection) const = 0;

			virtual void fft_2D_R2C(const Eigen::Tensor<DataType,2>&, Eigen::Tensor<DataType,1>, FFTDirection) const = 0;
			
			virtual void fft_2D_C2C(Eigen::Tensor<DataType,2>&, FFTDirection) const = 0;
			
			virtual void fft_2D_R2C(Eigen::Tensor<DataType,2>&, FFTDirection) const = 0;

			//--------------------------------
			//---------------ND---------------
			//--------------------------------
			virtual void fft_ND_C2C(const Eigen::Tensor<DataType, Rank>&, Eigen::Tensor<DataType, Rank>, FFTDirection) const = 0;
			
			virtual void fft_ND_R2C(const Eigen::Tensor<DataType, Rank>&, Eigen::Tensor<DataType, Rank>, FFTDirection) const = 0;

			virtual void fft_ND_C2C(Eigen::Tensor<DataType, Rank>&, FFTDirection) const = 0;

			virtual void fft_ND_R2C(Eigen::Tensor<DataType, Rank>&, FFTDirection) const = 0;

			virtual ~FFTStrategy() = default;
	};
}

#endif
