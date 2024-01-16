#ifndef SEQUENTIALFFT_3D_HPP
#define SEQUENTIALFFT_3D_HPP

#include <iostream>
#include "../../../FFTSolver.hpp"
#include "../../../utils/FFTUtils.hpp"

using namespace std;
namespace fftcore{

    template<typename FloatingType = double>
    class SequentialFFT_3D : public FFT_3D<FloatingType>
    {
    public:
        using typename FFT_3D<FloatingType>::RTensor_3D;
        using typename FFT_3D<FloatingType>::CTensor_3D;
        
        void fft(const CTensor_3D &, CTensor_3D &, FFTDirection) const;

        void fft(const RTensor_3D &, CTensor_3D &, FFTDirection) const;

        void fft(CTensor_3D &, FFTDirection) const;

        ~SequentialFFT_3D() = default;
    };

    template <typename FloatingType>
    void SequentialFFT_3D<FloatingType>::fft(const CTensor_3D &input, CTensor_3D &output, FFTDirection fftDirection) const
    {
        output = input; //deep copy
        fft(output, fftDirection);
    };

    template <typename FloatingType>
    void SequentialFFT_3D<FloatingType>::fft(const RTensor_3D &, CTensor_3D &, FFTDirection) const
    {
        throw NotSupportedException("Operation is not supported");
    };

    /**
     * This computes the FFT in 3 dimensions using Cooley-Tuckey algorithm. The method leverage on the .chip()
     * method of Eigen::tensor to get the correct view of the plane and then of the line to transform.
     *  
     * The algorithm follows the following  steps considering the orientation (x,y,z) of the tensor:
     * 1) For all the couples (y,z) compute all the one-dimensional structures along the x-dimension:
     * 2) For all the couples (x,z) compute all the one-dimensional structures along the y-dimension:
     * 3) For all the couples (x,y) compute all the one-dimensional structures along the z-dimension:
     *  
     * The 
     * @author Edoardo Carrà
     */
    template <typename FloatingType>
    void SequentialFFT_3D<FloatingType>::fft(CTensor_3D &input_output, FFTDirection fftDirection) const
    {
        using Complex   =   std::complex<FloatingType>;

        int num_planes, num_lines, num_elements;
        Eigen::Tensor<Complex, 2> plane;
        Eigen::Tensor<Complex, 1> line;  

        int log2n;
        Complex w, wm, t, u;
        int m, m2;

        
        const auto& dimensions  =   input_output.dimensions();
        const int X_DIMENSION   =   0,
                  Y_DIMENSION   =   1,
                  Z_DIMENSION   =   2;
                  
        const auto x_dimension_size =   dimensions[X_DIMENSION];
        const auto y_dimension_size =   dimensions[Y_DIMENSION];
        const auto z_dimension_size =   dimensions[Z_DIMENSION];

        //input dimensions check
        for(int i=0; i<3; i++){
            assert(!(dimensions[i] & (dimensions[i] - 1)) && "All dimensions of the input must be a power of 2.");
        }

        //conjugate if inverse
        if(fftDirection == FFT_INVERSE){            
            FFTUtils::conjugate(input_output);
        }

        num_planes = z_dimension_size; 

        // loop over each direction of the plane
        /* plane_dir = 0) For all the couples (y,z) compute all the one-dimensional structures along the x-dimension:
         * plane_dir = 1) For all the couples (x,z) compute all the one-dimensional structures along the y-dimension:
         */
        for(int plane_dir=0; plane_dir<2; plane_dir++){

            for(int plane_offset=0; plane_offset<num_planes; plane_offset++){
            
                auto plane = input_output.chip(plane_offset, Z_DIMENSION);

                num_elements    =   plane_dir==X_DIMENSION? x_dimension_size : y_dimension_size;
                num_lines       =   plane_dir==X_DIMENSION? y_dimension_size : x_dimension_size;
                
                log2n = std::log2(num_elements);

                // During the first iteration, the access pattern follows row-major order. 
                // Subsequent iterations adopt the column-major order, which is the standard 
                // format for Eigen tensors.            
                for(int line_offset=0; line_offset < num_lines; line_offset++){
                    
                    line=plane.chip(line_offset, plane_dir==X_DIMENSION? Y_DIMENSION : X_DIMENSION);
                

                    // Bit-reversal permutation
                    FFTUtils::bit_reversal_permutation(line);

                    
                    for (int s = 1; s <= log2n; ++s)
                    {
                        m   =   1 << s;  // 2 power s
                        m2  =   m >> 1; // m2 = m/2 -1
                        wm  =   exp(Complex(0, -2 * M_PI / m)); // w_m = e^(-2*pi/m)

                        for(int k = 0; k < num_elements; k += m)
                        {
                            w = Complex(1, 0);
                            for(int j = 0; j < m2; ++j)
                            {
                                t   =   w * line(k + j + m2);
                                u   =   line(k + j);

                                line(k + j + m2)    =   u - t;
                                line(k + j)         =   u + t;

                                w *= wm;
                            }
                        }
                    }//ONE-DIMENSIONAL FFT

                    // copy the line back in the original tensor 
                    if(plane_dir==X_DIMENSION){
                        for (Eigen::Index x = 0; x < x_dimension_size; ++x) {
                            input_output(x,line_offset,plane_offset)=line(x);
                        }
                    }else{
                        for (Eigen::Index y = 0; y < y_dimension_size; ++y) {
                            input_output(line_offset,y,plane_offset)=line(y);
                        }
                    }

                }//Computes all lines on a plane
            }//computes all planes in the cube
        }//switch direction on the planes


        log2n   =   std::log2(z_dimension_size);
        num_elements    =   z_dimension_size;

        // For all the couples (x,y) compute all the one-dimensional structures along the z-dimension:
        for(Eigen::Index x=0; x<x_dimension_size; x++){
            for(Eigen::Index y=0; y<y_dimension_size; y++){
            
                plane = input_output.chip(x,X_DIMENSION);
                // dim 0 = y and dim 1 = z
                line = plane.chip(y,0);
                                
                // Normal 1-D Cooley-Tukey FFT
                
                // Bit-reversal permutation
                FFTUtils::bit_reversal_permutation(line);


                Complex w, wm, t, u;
                int m, m2;
                for (int s = 1; s <= log2n; ++s)
                {
                    m = 1 << s;  // 2 power s
                    m2 = m >> 1; // m2 = m/2 -1
                    wm = exp(Complex(0, -2 * M_PI / m)); // w_m = e^(-2*pi/m)

                    for(int k = 0; k < num_elements; k += m)
                    {
                        w = Complex(1, 0);
                        for(int j = 0; j < m2; ++j)
                        {

                            t = w * line(k + j + m2);
                            u = line(k + j);

                            line(k + j + m2) = u - t;
                            line(k + j) = u + t;

                            w *= wm;
                        }
                    }
                }//ONE-DIMENSIONAL FFT

                // copy the line back in the original tensor 
                for (Eigen::Index z = 0; z < z_dimension_size; ++z) {
                    input_output(x,y,z)=line(z);
                }
                
            }//y
        }// x

        //re-conjugate and scale if inverse
        if(fftDirection == FFT_INVERSE){
            FFTUtils::conjugate(input_output);
            //scale
            input_output = input_output * Complex(1.0/(dimensions[0]*dimensions[1]*dimensions[2]), 0);
        }  
    };
}

#endif //SEQUENTIALFFT_3D_HPP