#ifndef _FFTSOLVER_H
#define _FFTSOLVER_H

#include "TensorFFTBase.hpp"
#include "FFTStrategy.hpp"
#include "FFTDataTypes.hpp"
#include <memory>
#include <type_traits>

namespace fftcore{	

	using fftcore::TensorFFTBase;
	template<typename DataType, int Rank>
	class FFTSolver{

		public:
			FFTSolver(std::unique_ptr<FFTStrategy<DataType, Rank>>&& strategy): _fftstrategy(std::move(strategy))
			{
				static_assert(std::is_same<DataType,double>::value || std::is_same<DataType,float>::value);
				static_assert(Rank>0);
			};
			
			void compute_fft_C2C(const TensorFFTBase<DataType, Rank>& input, TensorFFTBase<DataType, Rank>& output, FFTDirection dir) const
			{
				switch (Rank){
					case 1: {
						_fftstrategy->fft_1D_C2C(input.get_tensor(), output.get_tensor(), dir);
						break;
					}
					case 2:{
						_fftstrategy->fft_2D_C2C(input.get_tensor(), output.get_tensor(), dir);
						break;
					}
					default:{
						_fftstrategy->fft_ND_C2C(input.get_tensor(), output.get_tensor(), dir);
					}
				}
			};

			void compute_fft_R2C(const TensorFFTBase<DataType, Rank>& input, TensorFFTBase<DataType, Rank>& output, FFTDirection dir) const{};
			
			void compute_fft_C2C(TensorFFTBase<DataType, Rank>& input_output,FFTDirection dir) const{};

			void compute_fft_R2C(TensorFFTBase<DataType, Rank>& input_output, FFTDirection dir) const{};

			//void set_strategy(std::unique_ptr<fftcore::FFTEngine<DataType, Rank>>&&);
		private:
			std::unique_ptr<FFTStrategy<DataType, Rank>> _fftstrategy;
	};
}
#endif
