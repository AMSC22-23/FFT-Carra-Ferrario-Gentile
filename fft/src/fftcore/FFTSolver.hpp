#ifndef H_FFTSOLVER_H
#define H_FFTSOLVER_H

#include "TensorFFTBase.hpp"
#include "FFTStrategy.hpp"
#include "FFTDataTypes.hpp"
#include <memory>
#include <type_traits>

namespace fftcore{	

	template<typename DataType, int Rank>
	class FFTSolver{

		using CTensorBase = TensorFFTBase<std::complex<DataType>, Rank>;
		using RTensorBase = TensorFFTBase<DataType, Rank>;

		public:
			FFTSolver(std::unique_ptr<FFTStrategy<DataType, Rank>>&& strategy): _fftstrategy(std::move(strategy))
			{
				static_assert(std::is_same<DataType,double>::value || std::is_same<DataType,float>::value);
				static_assert(Rank>0 && Rank<=3, "Rank not supported");
			};
			
			void compute_fft(const CTensorBase& input, CTensorBase& output, FFTDirection dir) const
			{
				_fftstrategy->fft(input.get_tensor(), output.get_tensor(), dir);
			};
			
			void compute_fft(const RTensorBase& input, CTensorBase& output, FFTDirection dir) const
			{
				_fftstrategy->fft(input.get_tensor(), output.get_tensor(), dir);
			};

			void compute_fft(CTensorBase& input_output,FFTDirection dir) const{
				_fftstrategy->fft(input_output.get_tensor(), dir);
			};

			//void set_strategy(std::unique_ptr<fftcore::FFTEngine<DataType, Rank>>&&);
		private:
			std::unique_ptr<FFTStrategy<DataType, Rank>> _fftstrategy;
	};
}
#endif
