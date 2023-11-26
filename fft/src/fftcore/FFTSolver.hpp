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
			void compute_fft(const TensorFFT<DataType, Rank>&, TensorFFT<DataType, Rank>&) const; // virtual?

			void ifft(const TensorFFT<DataType, Rank>&,TensorFFT<DataType, Rank>&) const;
			
			void fft(TensorFFT<DataType, Rank>&) const;

			void ifft(TensorFFT<DataType, Rank>&) const;

			void set_strategy(std::unique_ptr<fftcore::FFTEngine<DataType, Rank>>&&);
		private:
			std::unique_ptr<fftcore::FFTEngine<DataType, Rank>> _fftengine;
	};
}
