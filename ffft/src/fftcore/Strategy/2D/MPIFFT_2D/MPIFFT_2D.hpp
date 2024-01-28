
#ifndef MPIFFT_2D_HPP
#define MPIFFT_2D_HPP

#include <iostream>
#include <mpi.h>
#include "../../../FFTSolver.hpp"
#include "../../../utils/FFTUtils.hpp"

using namespace std;

namespace fftcore{

    /**
     * @brief MPI implementation of the 2 dimensional FFT
     * 
     * MPIFFT_2D strategy simply executes the fft 
     * on every row of the input matrix, and then on every column, using
     * one dimensional MPIFFT strategy.
     * 
     * @author Daniele Ferrario 
    */
    template <typename FloatingType = double>
    class MPIFFT_2D : public FFT_2D<FloatingType>
    {
    public:
        using typename FFT_2D<FloatingType>::RTensor_2D;
        using typename FFT_2D<FloatingType>::CTensor_2D;

        // @TODO: Find a way to import from FFT_1D 
        using typename FFT_2D<FloatingType>::CTensor_1D;
        using typename FFT_2D<FloatingType>::RTensor_1D;
        
        void fft(const CTensor_2D &, CTensor_2D &, FFTDirection) const;

        void fft(const RTensor_2D &, CTensor_2D &, FFTDirection) const;

        void fft(CTensor_2D &, FFTDirection) const;

        ~MPIFFT_2D() = default;

    };

    /**
     * @brief Out-of-place of 2-D complex tensors (Matrices)
    */
    template <typename FloatingType>
    void MPIFFT_2D<FloatingType>::fft(const CTensor_2D &/*input*/, CTensor_2D &/*output*/, FFTDirection /*fftDirection*/) const
    {
        throw NotSupportedException("MPIFFT_2D doesn't support out-of-place 2D FFT");
    };

    /**
     * @brief Out-of-place Real to Complex of real 2-D tensors (Matrices)
    */
    template <typename FloatingType>
    void MPIFFT_2D<FloatingType>::fft(const RTensor_2D &/*input*/, CTensor_2D &/*output*/, FFTDirection/*fftDirection*/) const
    {
        throw NotSupportedException("MPIFFT_2D doesn't support out-of-place 2D FFT");
    };

    /**
     * @brief In-place fft of 2-D complex tensors (Matrices).
     * @see MPIFFT_2D for details. 
     * @param &global_tensor is the 2D Eigen Tensor to transform.
    */
    template <typename FloatingType>
    void MPIFFT_2D<FloatingType>::fft(CTensor_2D &global_tensor, FFTDirection fftDirection) const
    {
        
        MPIFFT<FloatingType> strategy;

        int rank, size;
        MPI_Datatype mpi_datatype = std::is_same<FloatingType, double>::value ? MPI_C_DOUBLE_COMPLEX : MPI_C_FLOAT_COMPLEX;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);


        // Retrieve the matrix dimensions
        const auto& dims = global_tensor.dimensions();
        
        // Rows
        CTensor_1D axis_tensor;
        for(int y=0; y<dims[0]; y++){
            // Curent 

            axis_tensor = global_tensor.chip(y,0);
            strategy.fft(axis_tensor, fftDirection);   
            if(rank == 0)
                global_tensor.chip(y,0) = axis_tensor;
        
        }
        MPI_Bcast(global_tensor.data(), global_tensor.size(), mpi_datatype, 0, MPI_COMM_WORLD);


        // Columns 
        for(int x=0; x<dims[1]; x++){
            axis_tensor= global_tensor.chip(x,1);
            strategy.fft(axis_tensor, fftDirection);    
            if(rank == 0)
                global_tensor.chip(x,1) = axis_tensor;
        }
        
        MPI_Bcast(global_tensor.data(), global_tensor.size(), mpi_datatype, 0, MPI_COMM_WORLD);

        
    }
}

#endif 
