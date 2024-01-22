#ifndef FFTSOLVER_HPP
#define FFTSOLVER_HPP

#include "Tensor/TensorFFTBase.hpp"
#include "Strategy/FFTStrategy.hpp"
#include "utils/FFTDataTypes.hpp"
#include "Timer/Timer.hpp"
#include <memory>
#include <type_traits>

namespace fftcore{	

	/**
	 * @brief Describes a solver for computing the FFT equipped with a particular strategy.
	 * 
	 * This class represents the context entity of the *strategy pattern* (see <a href="https://refactoring.guru/design-patterns/strategy">more</a>).
	 * It allows to encapsulate a particular strategy for computing the FFT and it is templated on the dimension and on the floating type. 
	 * 
	 * At the creation of the solver, the FFTSolver object is bounded to a strategy defined by the client. The FFTSolver and FFTStrategy are then related
	 * by *composition*, that is implemented thanks to the std::unique_ptr. 
	 * 
	 * An example of creation of a 2 dimensional  FFTSolver that uses the strategy SequentialFFT_2D using double as floating type is:
	 * `FFTSolver<2, double> solver(std::make_unique<SequentialFFT_2D<double>>());`
	 * 
	 * The FFTSolver contains also a Timer object that is used to measure the time taken by the strategy to compute the fft.
	 * 
	 * This design allows to introduce new strategies without having to change the context (open/closed principle). Moreover, 
	 * the implementation details of an algorithm are completly isolated from the code that uses it.
	 */
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
			
			/**
			* Complex to Complex out-of-place FFT. This is the most general case, in which no symmetry of the Fourier transform
			* can be used effectively.
			* 
			* @param [in] input a constant refeference to the input complex TensorFFTBase that won't be modified by the method.
			* @param [out] output a reference to the output complex TensorFFTBase that contains the transformed data.
			* @param [in] dir the fft direction.
			*/ 
			void compute_fft(const CTensorBase& input, CTensorBase& output, FFTDirection dir)
			{
				_timer.start();
				_fftstrategy->fft(input.get_tensor(), output.get_tensor(), dir);
				_timer.stop();
			};
			
			/**
			* Real to Complex out-of-place FFT. This method can save memory, since real numbers requires less memory.
			* This can be used also to exploit Fourier's transform symmetries to gain performance. 
			*
			* @param [in] input a constant refeference to the input real TensorFFTBase that won't be modified by the method.
			* @param [out] output a reference to the output complex TensorFFTBase that contains the transformed data.
			* @param [in] dir the fft direction.
			*/ 
			void compute_fft(const RTensorBase& input, CTensorBase& output, FFTDirection dir)
			{
				_timer.start();
				_fftstrategy->fft(input.get_tensor(), output.get_tensor(), dir);
				_timer.stop();
			};

			/**
			* In-place FFT. No conservation of the original data.
			*
			* @param [in,out] input_output a refeference to the input real TensorFFTBase that will contain the transformed data,
			*	loosing the original tensor.
			* @param [in] dir the fft direction.
			*/ 
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
			
		private:
			std::unique_ptr<FFTStrategy<Rank, FloatingType>> _fftstrategy;
			Timer _timer;
	};
}

#endif //FFTSOLVER_HPP
