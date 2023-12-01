#include "TensorFFTBase.hpp"
#include "FFTStrategy.hpp"
#include <memory>

namespace fftcore{	

	// Useful alias	
	template<typename DataType, int Rank>
	using TensorFFT = TensorFFTBase<DataType, Rank>;

	//template<typename DataType, int Rank>
	class FFTSolver{
		public:
			FFTSolver(){};
			
			template<typename DataType, int Rank>
			void compute_fft(const TensorFFT& input, TensorFFT& output) const;

			template<typename DataType, int Rank>
			void compute_ifft(const TensorFFT& input, TensorFFT& output) const;
			
			template<typename DataType, int Rank>
			void compute_fft(TensorFFT& input_output) const;

			template<typename DataType, int Rank>
			void compute_ifft(TensorFFT& input_output) const;

			//void set_strategy(std::unique_ptr<fftcore::FFTEngine<DataType, Rank>>&&);
		private:
			std::unique_ptr<fftcore::FFTStrategy> _fftengine;
	};
}
