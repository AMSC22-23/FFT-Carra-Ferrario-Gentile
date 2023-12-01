#include <unsupported/Eigen/CXX11/Tensor>

namespace fftcore{

	template<typename DataType, int Rank>
	using ETensor = Eigen::Tensor<DataType, Rank>;
	
	class FFTEngine{
		public:
			FFTEngine(){};
			
			//--------------------------------
			//---------------1D---------------
			//--------------------------------

			template<typename DataType>
			virtual void fft_1D(const Eigen::Tensor<DataType,1>&, Eigen::Tensor<DataType,1>) const = 0;

			template<typename DataType>
			virtual void ifft_1D(const Eigen::Tensor<DataType,1>&, Eigen::Tensor<DataType,1>) const = 0;

			template<typename DataType>
			virtual void fft_1D(Eigen::Tensor<DataType,1>&) const = 0;

			template<typename DataType>
			virtual void ifft_1D(Eigen::Tensor<DataType,1>&) const = 0;

			//--------------------------------
			//---------------2D---------------
			//--------------------------------
			template<typename DataType>
			virtual void fft_2D(const Eigen::Tensor<DataType,2>&, Eigen::Tensor<DataType,2>) const = 0;

			template<typename DataType>
			virtual void ifft_2D(const Eigen::Tensor<DataType,2>&, Eigen::Tensor<DataType,2>) const = 0;

			template<typename DataType>
			virtual void fft_2D(Eigen::Tensor<DataType,2>&) const = 0;

			template<typename DataType>
			virtual void ifft_2D(Eigen::Tensor<DataType,2>&) const = 0;

			//--------------------------------
			//---------------ND---------------
			//--------------------------------
			template<typename DataType, int Rank>
			virtual void fft_ND(const Eigen::Tensor<DataType, Rank>&, Eigen::Tensor<DataType, Rank>) const = 0;
			
			template<typename DataType, int Rank>
			virtual void ifft_ND(const Eigen::Tensor<DataType, Rank>&, Eigen::Tensor<DataType, Rank>) const = 0;

			template<typename DataType, int Rank>
			virtual void fft_ND(Eigen::Tensor<DataType, Rank>&) const = 0;

			template<typename DataType, int Rank>
			virtual void ifft_ND(Eigen::Tensor<DataType, Rank>&) const = 0;

			virtual ~FFTEngine() = default; // da rivedere
	};
}

