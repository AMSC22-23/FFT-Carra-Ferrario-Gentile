#include "FFTSolver.hpp"
#include <iostream>
#include <memory>

using namespace fftcore;


template<typename DataType, int Rank>
class SequentialFFT:
public FFT_1D<DataType>{
//public FFT_2D<DataType> {
        //using CTensor_2D;

		public:
			void fft(const Eigen::Tensor<std::complex<DataType>, 1>& , Eigen::Tensor<std::complex<DataType>, 1>&, FFTDirection) const {
                std::cout<<"fft 1-d C-C out-of-place"<<std::endl;
            };
			
			void fft(const Eigen::Tensor<DataType, 1>&, Eigen::Tensor<std::complex<DataType>, 1>&, FFTDirection) const {
                std::cout<<"fft 1-d R-C out-of-place"<<std::endl;
            };

			void fft(Eigen::Tensor<std::complex<DataType>, 1>&, FFTDirection) const {
                std::cout<<"fft 1-d C-C in-place"<<std::endl;
            };
			
			/*virtual void fft(const CTensor_2D&, CTensor_2D&, FFTDirection) const {
                std::cout<<"fft 2-d C-C out-of-place"<<std::endl;
            };
			
			virtual void fft(const RTensor_2D&, CTensor_2D&, FFTDirection) const {
                std::cout<<"fft 2-d R-C out-of-place"<<std::endl;

            };

			virtual void fft(CTensor_2D&, FFTDirection) const {
                std::cout<<"fft 2-d in-place"<<std::endl;
            };*/

			~SequentialFFT() = default;
};

int main(){
    FFTSolver<double,1> fft_solver(std::make_unique<SequentialFFT<double,1>>());
    TensorFFTBase<std::complex<double>,1> tensor_out(10);
    TensorFFTBase<std::complex<double>,1> tensor_in(10);
    //operazioni su w
    fft_solver.compute_fft(tensor_in, tensor_out, FFT_FORWARD);
}