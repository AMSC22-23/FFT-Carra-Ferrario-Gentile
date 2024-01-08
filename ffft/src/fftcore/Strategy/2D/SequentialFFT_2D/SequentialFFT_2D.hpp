#ifndef SEQUENTIALFFT_2D_HPP
#define SEQUENTIALFFT_2D_HPP


#include <iostream>
#include "../../../FFTSolver.hpp"
#include "../../../utils/FFTUtils.hpp"

using namespace std;
namespace fftcore{

    template<typename FloatingType = double>
    class SequentialFFT_2D : public FFT_2D<FloatingType>
    {
    public:
        using typename FFT_2D<FloatingType>::RTensor_2D;
        using typename FFT_2D<FloatingType>::CTensor_2D;
        
        void fft(const CTensor_2D &, CTensor_2D &, FFTDirection) const;

        void fft(const RTensor_2D &, CTensor_2D &, FFTDirection) const;

        void fft(CTensor_2D &, FFTDirection) const;

        ~SequentialFFT_2D() = default;
    };

    template <typename FloatingType>
    void SequentialFFT_2D<FloatingType>::fft(const CTensor_2D &input, CTensor_2D &output, FFTDirection fftDirection) const
    {
        output = input; //deep copy
        fft(output, fftDirection);
    };

    template <typename FloatingType>
    void SequentialFFT_2D<FloatingType>::fft(const RTensor_2D &, CTensor_2D &, FFTDirection) const
    {
        throw NotSupportedException("Operation is not supported");
    };

    /**
     * This compute FFT in 2 dimensions using Cooley-Tuckey algorithm. The algorithm follows the following
     * steps:
     * - First, the rows are transformed using the Cooley-Tuckey algorithm for one dimensional data. This first
     *   step is performed in row-major order.
     * - Second, the columns are transformed, accessing the elements in column-major order.
     *  
     * @author Edoardo Carrà
     */
    template <typename FloatingType>
    void SequentialFFT_2D<FloatingType>::fft(CTensor_2D &input_output, FFTDirection fftDirection) const
    {
        int row, col, row1, col1;

        using Complex = std::complex<FloatingType>;
        const auto& dimensions = input_output.dimensions();

        //input dimensions check
        for(int i=0; i<2; i++){
            assert(!(dimensions[i] & (dimensions[i] - 1)) && "All dimensions of the input must be a power of 2.");
        }

        //conjugate if inverse
        if(fftDirection == FFT_INVERSE){            
            FFTUtils::conjugate(input_output);
        }

        // loop over each dimension
        for(int dim=0; dim<2; dim++){
            int log2n = std::log2(dimensions[dim]);

            // During the first iteration, the access pattern follows row-major order. 
            // Subsequent iterations adopt the column-major order, which is the standard 
            // format for Eigen tensors.            
            for(int offset=0; offset< dimensions[dim==0?1:0]; offset++ ){
                
                // Normal 1-D Cooley-Tukey FFT

                // Bit-reversal permutation
                for (int i = 0; i < dimensions[dim]; ++i)
                {
                    int rev = FFTUtils::reverseBits(i, log2n);
                    if (i < rev)
                    {
                        // In each iteration, one dimension will be accessed by offset, 
                        // while the other dimension undergoes a standard Fast Fourier Transform (FFT).
                        row = dim==1? offset : i;
                        col = dim==0? offset : i; 

                        row1 = dim==1? offset : rev;
                        col1 = dim==0? offset : rev; 

                        std::swap(input_output(row,col),input_output(row1,col1));
                    }
                }

                Complex w, wm, t, u;
                int m, m2;
                for (int s = 1; s <= log2n; ++s)
                {
                    m = 1 << s;  // 2 power s
                    m2 = m >> 1; // m2 = m/2 -1
                    wm = exp(Complex(0, -2 * M_PI / m)); // w_m = e^(-2*pi/m)

                    for(int k = 0; k < dimensions[dim]; k += m)
                    {
                        w = Complex(1, 0);
                        for(int j = 0; j < m2; ++j)
                        {

                            // In each iteration, one dimension will be accessed by offset, 
                            // while the other dimension undergoes a standard Fast Fourier Transform (FFT).
                            row = dim==1? offset : k + j + m2;
                            col = dim==0? offset : k + j + m2; 
                            
                            row1 = dim==1? offset : k + j;
                            col1 = dim==0? offset : k + j; 

                            t = w * input_output(row,col);
                            u = input_output(row1,col1);

                            input_output(row1,col1) = u + t;
                            input_output(row,col) = u - t;

                            w *= wm;
                        }
                    }
                }
            }
        }
        
        //re-conjugate and scale if inverse
        if(fftDirection == FFT_INVERSE){
            FFTUtils::conjugate(input_output);
            //scale
            input_output = input_output * Complex(1.0/(dimensions[0]*dimensions[1]), 0);
        }  
    };
}

#endif //SEQUENTIALFFT_2D_HPP