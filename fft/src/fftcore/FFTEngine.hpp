#include <unsupported/Eigen/CXX11/Tensor>

namespace fftcore{

	template<typename DataType, int Rank>
	using EigenTensor =  Eigen::Tensor<DataType, Rank>;

	template<typename DataType, int Rank>
	class FFTEngine{
		public:
			virtual FFTEngine(){};
			virtual void fft(const &EigenTensor, &EigenTensor) const = 0;
			virtual void ifft(const &EigenTensor, &EigenTensor) const = 0;
			virtual EigenTensor& fft(EigenTensor) const = 0;
			virtual EigenTensor& ifft(EigenTensor) const = 0;
			
			virtual ~FFTEngine() = default; // da rivedere
	};
}

