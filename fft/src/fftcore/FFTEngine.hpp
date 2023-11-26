#include <unsupported/Eigen/CXX11/Tensor>

namespace fftcore{

	template<typename DataType, int Rank>
	using ETensor = Eigen::Tensor<DataType, Rank>;

	template<typename DataType, int Rank>
	class FFTEngine{
		public:
			FFTEngine(){};
			virtual void fft(const ETensor<DataType, Rank>&, ETensor<DataType, Rank>) const = 0;
			virtual void ifft(const ETensor<DataType, Rank>&, ETensor<DataType, Rank>&) const = 0;
			virtual ETensor<DataType, Rank>& fft(ETensor<DataType, Rank>) const = 0;
			virtual ETensor<DataType, Rank>& ifft(ETensor<DataType, Rank>) const = 0;
			
			virtual ~FFTEngine() = default; // da rivedere
	};
}

