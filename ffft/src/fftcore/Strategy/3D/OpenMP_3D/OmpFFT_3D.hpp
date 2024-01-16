#ifndef OMP_FFT_3D_HPP
#define OMP_FFT_3D_HPP

#include <iostream>
#include <omp.h>
#include "../../../FFTSolver.hpp"
#include "../../../utils/FFTUtils.hpp"

using namespace std;
namespace fftcore{

    template<typename FloatingType = double>
    class OmpFFT_3D : public FFT_3D<FloatingType>
    {
    public:
        using typename FFT_3D<FloatingType>::RTensor_3D;
        using typename FFT_3D<FloatingType>::CTensor_3D;
        
        void fft(const CTensor_3D &, CTensor_3D &, FFTDirection) const;

        void fft(const RTensor_3D &, CTensor_3D &, FFTDirection) const;

        void fft(CTensor_3D &, FFTDirection) const;

        ~OmpFFT_3D() = default;
    };

    template <typename FloatingType>
    void OmpFFT_3D<FloatingType>::fft(const CTensor_3D &input, CTensor_3D &output, FFTDirection fftDirection) const
    {
        output = input; //deep copy
        fft(output, fftDirection);
    };

    template <typename FloatingType>
    void OmpFFT_3D<FloatingType>::fft(const RTensor_3D &, CTensor_3D &, FFTDirection) const
    {
        throw NotSupportedException("Operation is not supported");
    };

    /**
     * This computes the FFT in 3 dimensions using Cooley-Tuckey algorithm. The code may be tortuous, but 
     * does not require any kind of overlay to acess data. The algorithm follows the following
     * steps considering the orientation (x,y,z) of the tensor:
     * 1) For all the couples (y,z) compute all the one-dimensional structures along the x-dimension:
     * 2) For all the couples (x,z) compute all the one-dimensional structures along the y-dimension:
     * 3) For all the couples (x,y) compute all the one-dimensional structures along the z-dimension:
     *  
     * Each one-dimensional FFT is computed independently by one single thread.
     * 
     * @author Edoardo Carrà
     */
    template <typename FloatingType>
    void OmpFFT_3D<FloatingType>::fft(CTensor_3D &input_output, FFTDirection fftDirection) const
    {
        using Complex = std::complex<FloatingType>;

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
        
        #pragma omp parallel
        {
            int num_planes, num_lines, num_elements;
            int x, y, z , x1, y1, z1;
            int log2n;

            Complex w, wm, t, u;
            int m, m2;

            // cube_dir = 0 -> 2D transform of all x-y cube's planes 
            for(int cube_dir=0; cube_dir<2; cube_dir++){
                num_planes  =   cube_dir==0? z_dimension_size : x_dimension_size; 

                // 2D-tranform of the plane
                for(int plane_dir=0; plane_dir<2; plane_dir++){
                    
                    /* cube_dir = 0 && plane_dir = 0 transform lines orthogonal to the y-z plane
                    * cube_dir = 0 && plane_dir = 1 transform lines orthogonal to the x-z plane
                    * cube_dir = 1 && plane_dir = 0 transform lines orthogonal to the x-y plane
                    */

                    for(int plane=0; plane<num_planes; plane++){

                        // skip this case which performs the same
                        if(cube_dir==1 && plane_dir==1)continue;

                        if(plane_dir==0){
                            num_elements    =   cube_dir==0? x_dimension_size : z_dimension_size;
                            num_lines       =   y_dimension_size;
                        }
                        else if(cube_dir==0){
                            num_elements    =   y_dimension_size;
                            num_lines       =   x_dimension_size;
                        }else{
                            num_elements    =   y_dimension_size;
                            num_lines       =   z_dimension_size;
                        }

                        log2n = std::log2(num_elements);

                        // During the first iteration, the access pattern follows row-major order. 
                        // Subsequent iterations adopt the column-major order, which is the standard 
                        // format for Eigen tensors.          
                        #pragma omp for
                        for(int line=0; line < num_lines; line++ ){
                            
                            // 1-D Cooley-Tukey FFT

                            // Bit-reversal permutation
                            for (int i = 0; i < num_elements; ++i)
                            {
                                int rev = FFTUtils::reverseBits(i, log2n);
                                if (i < rev)
                                {
                                    if(cube_dir==0 && plane_dir==0){
                                        x = i;
                                        x1 = rev;
                                        y = y1 = line;
                                        z = z1 = plane;
                                    }else if(cube_dir==0 && plane_dir==1){
                                        y = i;
                                        y1 = rev;
                                        x = x1 = line;
                                        z = z1 = plane;
                                    }else if(cube_dir==1 && plane_dir==0){
                                        z = i;
                                        z1 = rev;
                                        x = x1 = plane;
                                        y = y1 = line;
                                    }

                                    std::swap(input_output(x,y,z),input_output(x1,y1,z1));
                                }
                            }

                            
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
                                        if(cube_dir==0 && plane_dir==0){
                                            x = k + j + m2;
                                            x1 = k + j;
                                            y = y1 = line;
                                            z = z1 = plane;
                                        }else if(cube_dir==0 && plane_dir==1){
                                            y = k + j + m2;
                                            y1 = k + j;
                                            x = x1 = line;
                                            z = z1 = plane;
                                        }else if(cube_dir==1 && plane_dir==0){
                                            z = k + j + m2;
                                            z1 = k + j;
                                            x = x1 = plane;
                                            y = y1 = line;
                                        }

                                        t = w * input_output(x,y,z);
                                        u = input_output(x1,y1,z1);

                                        input_output(x,y,z) = u - t;
                                        input_output(x1,y1,z1) = u + t;

                                        w *= wm;
                                    }
                                }
                            }//ONE-DIMENSIONAL FFT
                        } // loop over the lines of one single plane - IMPLICIT BARRIER
                    }// loop over the lines of all the planes
                } // switch plane.
            }
        } // END PARALLEL REGION
        
        //re-conjugate and scale if inverse
        if(fftDirection == FFT_INVERSE){
            FFTUtils::conjugate(input_output);
            //scale
            input_output = input_output * Complex(1.0/(x_dimension_size * y_dimension_size * z_dimension_size), 0);
        }  
    };
}

#endif //OMP_FFT_3D_HPP