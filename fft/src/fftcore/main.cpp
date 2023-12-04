#include "FFTSolver.hpp"
#include <iostream>
#include <memory>

using namespace fftcore;

template<typename DataType>
class SequentialFFT:
public FFT_1D<DataType>,
public FFT_2D<DataType>, 
public FFT_3D<DataType>{
		public:
			void fft(const CTensor_1D<DataType>& , CTensor_1D<DataType>&, FFTDirection) const {
                std::cout<<"fft 1-d C-C out-of-place"<<std::endl;
            };
			
			void fft(const RTensor_1D<DataType>&, CTensor_1D<DataType>&, FFTDirection) const {
                std::cout<<"fft 1-d R-C out-of-place"<<std::endl;
            };

			void fft(CTensor_1D<DataType>&, FFTDirection) const {

                std::cout<<"fft 1-d C-C in-place"<<std::endl;
            };
			
			virtual void fft(const CTensor_2D<DataType>&, CTensor_2D<DataType>&, FFTDirection) const {
                std::cout<<"fft 2-d C-C out-of-place"<<std::endl;
            };
			
			virtual void fft(const RTensor_2D<DataType>&, CTensor_2D<DataType>&, FFTDirection) const {
                std::cout<<"fft 2-d R-C out-of-place"<<std::endl;

            };

			virtual void fft(CTensor_2D<DataType>&, FFTDirection) const {
                std::cout<<"fft 2-d in-place"<<std::endl;
            };

			virtual void fft(const CTensor_3D<DataType>&, CTensor_3D<DataType>&, FFTDirection) const {
                std::cout<<"fft 3-d C-C out-of-place"<<std::endl;
            };
			
			virtual void fft(const RTensor_3D<DataType>&, CTensor_3D<DataType>&, FFTDirection) const {
                std::cout<<"fft 3-d R-C out-of-place"<<std::endl;

            };

			virtual void fft(CTensor_3D<DataType>&, FFTDirection) const {
                std::cout<<"fft 3-d in-place"<<std::endl;
            };

			~SequentialFFT() = default;
};

int main(){
    {
        FFTSolver<double,1> fft_solver(std::make_unique<SequentialFFT<double>>());
        TensorFFTBase<std::complex<double>,1> tensor_out_complex(10);
        TensorFFTBase<std::complex<double>,1> tensor_in_complex(10);
        TensorFFTBase<double,1> tensor_in_real(10);
        
        fft_solver.compute_fft(tensor_in_complex, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_real, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_complex, FFT_FORWARD);

    } // 1D 

    {
        FFTSolver<double,2> fft_solver(std::make_unique<SequentialFFT<double>>());
        TensorFFTBase<std::complex<double>,2> tensor_out_complex(10,10);
        TensorFFTBase<std::complex<double>,2> tensor_in_complex(10,10);
        TensorFFTBase<double,2> tensor_in_real(10,10);
        
        fft_solver.compute_fft(tensor_in_complex, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_real, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_complex, FFT_FORWARD);

    }// 2D
    {
        FFTSolver<double,3> fft_solver(std::make_unique<SequentialFFT<double>>());
        TensorFFTBase<std::complex<double>,3> tensor_out_complex(10,10,10);
        TensorFFTBase<std::complex<double>,3> tensor_in_complex(10,10,10);
        TensorFFTBase<double,3> tensor_in_real(10,10,10);
        
        fft_solver.compute_fft(tensor_in_complex, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_real, tensor_out_complex, FFT_FORWARD);
        fft_solver.compute_fft(tensor_in_complex, FFT_FORWARD);

    }// 3D


    
}