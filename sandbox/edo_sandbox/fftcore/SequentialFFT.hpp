#ifndef H_SEQUENTIALFFT_H
#define H_SEQUENTIALFFT_H

#include "FFTStrategy.hpp"
#include <iostream>

namespace fftcore{
	
	template<typename DataType, int Rank>
	class SequentialFFT : public FFTStrategy<DataType, Rank>{
		public:

			SequentialFFT(){};

			//--------------------------------
			//---------------1D---------------
			//--------------------------------
			void fft_1D_C2C(const Eigen::Tensor<DataType,1>&, Eigen::Tensor<DataType,1>, FFTDirection)  const
			{
				std::cout << "miao" << std::endl;
			};

			void fft_1D_R2C(const Eigen::Tensor<DataType,1>&, Eigen::Tensor<DataType,1>, FFTDirection)  const 
			{
				std::cout << "miao" << std::endl;
			};
			
			void fft_1D_C2C(Eigen::Tensor<DataType,1>&, FFTDirection)  const 
			{
				std::cout << "miao" << std::endl;
			};
			
			void fft_1D_R2C(Eigen::Tensor<DataType,1>&, FFTDirection)  const
			{
				std::cout << "miao" << std::endl;
			};

			//--------------------------------
			//---------------2D---------------
			//--------------------------------			
			void fft_2D_C2C(const Eigen::Tensor<DataType,2>&, Eigen::Tensor<DataType,1>, FFTDirection)  const 
			{
				std::cout << "miao" << std::endl;
			};

			void fft_2D_R2C(const Eigen::Tensor<DataType,2>&, Eigen::Tensor<DataType,1>, FFTDirection)  const 
			{
				std::cout << "miao" << std::endl;
			};
			
			void fft_2D_C2C(Eigen::Tensor<DataType,2>&, FFTDirection)  const 
			{
				std::cout << "miao" << std::endl;
			};
			
			void fft_2D_R2C(Eigen::Tensor<DataType,2>&, FFTDirection)  const 
			{
				std::cout << "miao" << std::endl;
			};

			//--------------------------------
			//---------------ND---------------
			//--------------------------------
			void fft_ND_C2C(const Eigen::Tensor<DataType, Rank>&, Eigen::Tensor<DataType, Rank>, FFTDirection)  const
			{
				std::cout << "miao" << std::endl;
			};
			
			void fft_ND_R2C(const Eigen::Tensor<DataType, Rank>&, Eigen::Tensor<DataType, Rank>, FFTDirection)   const 
			{
				std::cout << "miao" << std::endl;
			};

			void fft_ND_C2C(Eigen::Tensor<DataType, Rank>&, FFTDirection)  const
			{
				std::cout << "miao" << std::endl;
			};

			void fft_ND_R2C(Eigen::Tensor<DataType, Rank>&, FFTDirection)  const 
			{
				std::cout << "miao" << std::endl;
			};

			 ~SequentialFFT() = default;
	};
}

#endif
