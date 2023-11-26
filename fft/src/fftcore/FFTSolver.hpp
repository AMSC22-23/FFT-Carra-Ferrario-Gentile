#include "TensorFFTBase.hpp"
#include "FFTEngine.hpp"
#include <memory>

namespace fftcore{
	

	// Useful alias
	
	template<typename DataType, int Rank>
	using TensorFFT = TensorFFTBase<DataType, Rank>;

	/**
	 * @TODO: Templating the entire class or just the methods?
	 * @TODO: Add to tenplate the FFTEngine strategy?
	*/
	template<typename DataType, int Rank>
	class FFTSolver{
		public:
			FFTSolver(){};
			void compute_fft(const &TensorFFT, &TensorFFT) const; // virtual?

			void ifft(const& TensorFFT,&TensorFFT) const;
			
			void fft(&TensorFFT) const;

			void ifft(&TensorFFT) const;

			void set_strategy(&&std::unique_ptr<FFTEngine>);
		private:
			std::unique_ptr<FFTEngine> _fftengine;
	};
}
