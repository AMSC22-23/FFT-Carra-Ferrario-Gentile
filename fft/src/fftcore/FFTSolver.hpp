#ifndef H_FFTSOLVER_H
#define H_FFTSOLVER_H

#include "TensorFFTBase.hpp"
#include "FFTStrategy.hpp"
#include "FFTDataTypes.hpp"
#include "Timer.hpp"
#include <memory>
#include <type_traits>

namespace fftcore{	

	template<int Rank, typename FloatingType = double>
	class FFTSolver{

		using CTensorBase = TensorFFTBase<std::complex<FloatingType>, Rank>;
		using RTensorBase = TensorFFTBase<FloatingType, Rank>;

		public:
			FFTSolver(std::unique_ptr<FFTStrategy<Rank, FloatingType>>&& strategy): _fftstrategy(std::move(strategy))
			{
				static_assert(std::is_floating_point<FloatingType>::value, "DataType must be floating point");
				static_assert(Rank>0 && Rank<=3, "Rank not supported");
			};
			
			void compute_fft(const CTensorBase& input, CTensorBase& output, FFTDirection dir)
			{
				_timer.start();
				_fftstrategy->fft(input.get_tensor(), output.get_tensor(), dir);
				_timer.stop();
			};
			
			void compute_fft(const RTensorBase& input, CTensorBase& output, FFTDirection dir)
			{
				_timer.start();
				_fftstrategy->fft(input.get_tensor(), output.get_tensor(), dir);
				_timer.stop();
			};

			void compute_fft(CTensorBase& input_output, FFTDirection dir)
			{
				_timer.start();
				_fftstrategy->fft(input_output.get_tensor(), dir);
				_timer.stop();
			};

			const Timer& get_timer() const
			{
				return _timer;
			};

			//void set_strategy(std::unique_ptr<fftcore::FFTEngine<DataType, Rank>>&&);
		private:
			std::unique_ptr<FFTStrategy<Rank, FloatingType>> _fftstrategy;
			Timer _timer;
	};
}
#endif
