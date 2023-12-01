#ifndef _FFTSOLVER_H
#define _FFTSOLVER_H

#include "TensorFFTBase.hpp"
#include "FFTStrategy.hpp"
#include "FFTDataTypes.hpp"
#include <memory>

namespace fftcore{	


	template<typename DataType, int Rank>
	class FFTSolver{
		
		using TensorFFT = TensorFFTBase<DataType, Rank>;

		public:
			FFTSolver(){};
			
			void compute_fft_C2C(const TensorFFT& input, TensorFFT& output, FFTDirection) const;

			void compute_fft_R2C(const TensorFFT& input, TensorFFT& output, FFTDirection) const;
			
			void compute_fft_C2C(TensorFFT& input_output, FFTDirection) const;

			void compute_fft_R2C(TensorFFT& input_output, FFTDirection) const;

			//void set_strategy(std::unique_ptr<fftcore::FFTEngine<DataType, Rank>>&&);
		private:
			std::unique_ptr<fftcore::FFTStrategy> _fftengine;
	};
}
#endif
