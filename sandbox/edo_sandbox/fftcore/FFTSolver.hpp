#ifndef _FFTSOLVER_H
#define _FFTSOLVER_H

#include "TensorFFTBase.hpp"
#include "FFTStrategy.hpp"
#include "FFTDataTypes.hpp"
#include <memory>
#include <type_traits>

namespace fftcore{	


	template<typename DataType, int Rank>
	class FFTSolver{
		
		using TensorFFT = TensorFFTBase<DataType, Rank>;

		public:
			FFTSolver(std::unique_ptr<FFTStrategy<DataType, Rank>>&& strategy): _fftstrategy(std::move(strategy))
			{
				static_assert(std::is_same<DataType,double>::value || std::is_same<DataType,float>::value);
				static_assert(Rank>0);
			};
			
			void compute_fft_C2C(const TensorFFT& input, TensorFFT& output, FFTDirection dir) const
			{
				switch (Rank){
					case 1: {
						_fftstrategy->fft_1D_C2C(input._tensor, output._tensor, dir);
						break;
					}
					case 2:{
						_fftstrategy->fft_2D_C2C(input._tensor, output._tensor, dir);
						break;
					}
					default:{
						_fftstrategy->fft_ND_C2C(input._tensor, output._tensor, dir);
					}
				}
			};

			void compute_fft_R2C(const TensorFFT& input, TensorFFT& output, FFTDirection) const{};
			
			void compute_fft_C2C(TensorFFT& input_output, FFTDirection) const{};

			void compute_fft_R2C(TensorFFT& input_output, FFTDirection) const{};

			//void set_strategy(std::unique_ptr<fftcore::FFTEngine<DataType, Rank>>&&);
		private:
			std::unique_ptr<FFTStrategy<DataType, Rank>> _fftstrategy;
	};
}
#endif
