#ifndef OMPFFT_HPP
#define OMPFFT_HPP

#include <iostream>
#include <omp.h>
#include <cmath>
#include "../../../FFTSolver.hpp"
#include "../../../utils/FFTUtils.hpp"

namespace fftcore{

    template<typename FloatingType = double>
    class OmpFFT:
    public FFT_1D<FloatingType>{
            public:
                using typename FFT_1D<FloatingType>::RTensor_1D;
                using typename FFT_1D<FloatingType>::CTensor_1D;


                void fft(const CTensor_1D& , CTensor_1D&, FFTDirection) const;
                
                void fft(const RTensor_1D&, CTensor_1D&, FFTDirection) const;

                void fft(CTensor_1D&, FFTDirection) const;

                ~OmpFFT() = default;
    };


    template<typename FloatingType>
    void OmpFFT<FloatingType>::fft(const CTensor_1D& input, CTensor_1D& output, FFTDirection) const {
        throw NotSupportedException("Operation is not supported");
    };

    template<typename FloatingType>
    void OmpFFT<FloatingType>::fft(const RTensor_1D&, CTensor_1D&, FFTDirection) const {
        throw NotSupportedException("Operation is not supported");
    };

    /**
     * Openmp in-place implementation of Cooley-Tuckey algorithm 
     * @author: Edoardo Carra
    */
    template<typename FloatingType>
    void OmpFFT<FloatingType>::fft(CTensor_1D& io_tensor, FFTDirection fft_direction) const {

#ifndef _OPENMP
        std::cerr<< "[WARNING] OMP not found, normal execution.";
#endif

        using Complex = std::complex<FloatingType>;
        // dimension of the 
        const Eigen::Index n = io_tensor.size();
        Complex w, wm, t, u;
        Eigen::Index m, m2, rev;
        const unsigned int log2n = std::log2(n);

        assert(!(n & (n - 1)) && "FFT length must be a power of 2.");


        //conjugate if inverse
        if(fft_direction == FFT_INVERSE){
            FFTUtils::conjugate(io_tensor);
        }


        #pragma omp parallel private(m, m2, w, t, u, wm, rev)
        {
#ifdef _OPENMP
            int omp_tn = omp_get_num_threads();
            assert(!(omp_tn & (omp_tn-1)) && "Number of threads must be a power of 2.");
#endif

            /* bit-reversal -> embarassingly parallel, exclusive read and
            * write access to the tensor. 
            */
            #pragma omp for
            for (Eigen::Index i = 0; i < n; ++i)
            {
                rev = FFTUtils::reverseBits(i, log2n);
                if (i < rev)
                {
                    std::swap(io_tensor[i], io_tensor[rev]);
                }
            }//implicit barrier here, no need to synchronize

            /*
            * F_1 : outer loop that executes log(n) iterations (stage). Loop-carried data dependence
            */
            for (unsigned int s = 1; s <= log2n; ++s) {
                m = 1 << s;         // 2 power s
                m2 = m >> 1;        // m2 = m/2 -1
                wm = exp(Complex(0,-2*M_PI/m));

                /*
                * F_2 : this loop is responsible to the creation of the block partition
                *       of the input vector. All threads are used until stage s is less then 
                *       log(n)-log(omp_get_num_threads()). After this threshold is reached, the
                *       number of threads used is halved at each stage. Loop independent data
                *       dependence.
                */
                #pragma omp for
                for (Eigen::Index k = 0; k < n; k += m) {
                    w=Complex(1,0);		    
                    
                    /*
                    *  F_3 : This loop performs the actual computation on each block partition with
                            a butterfly computation. Loop independent data dependence
                    */
                    #pragma omp simd
                    for (Eigen::Index j = 0; j < m2; ++j) {

                        t = w * io_tensor[j + k + m2];
                        u = io_tensor[j + k];
                        
                        // butterfly
                        io_tensor[k+j] = u + t;
                        io_tensor[k + m2 + j] = u - t;

                        w=w*wm; // root of unity
                    } // F_3 simd
                }// F_2 implicit barrier here at the end of the stage
            }//F_1
        } //end parallel region

       //re-conjugate and scale if inverse
        if(fft_direction == FFT_INVERSE){
            FFTUtils::conjugate(io_tensor);
            io_tensor = io_tensor * Complex(1.0/n, 0);
        }         
    };

}


#endif //OMPFFT_HPP
