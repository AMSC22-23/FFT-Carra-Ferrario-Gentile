
#ifndef MPIFFT_2D_HPP
#define MPIFFT_2D_HPP

#include "../../../FFTSolver.hpp"
#include "../../../utils/FFTUtils.hpp"
#include "../../FFTStrategy.hpp"
#include "../../1D/MPIFFT/MPIFFT.hpp"
#include <iostream>
#include <mpi.h>

using namespace fftcore;
using namespace std;

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
void MPIFFT_2D<FloatingType>::fft(const CTensor_2D &input, CTensor_2D &output, FFTDirection) const
{
    throw NotSupportedException("Operation is not supported");
};

/**
 * @brief Out-of-place Real to Complex of real 2-D tensors (Matrices)
*/
template <typename FloatingType>
void MPIFFT_2D<FloatingType>::fft(const RTensor_2D &, CTensor_2D &, FFTDirection) const
{
    throw NotSupportedException("Operation is not supported");
};

/**
 * @brief In-place fft of 2-D complex tensors (Matrices).
 * @see MPIFFT_2D for details. 
 * @param &global_tensor is the 2D Eigen Tensor to transform.
*/
template <typename FloatingType>
void MPIFFT_2D<FloatingType>::fft(CTensor_2D &global_tensor, fftcore::FFTDirection fftDirection) const
{
    
    MPIFFT strategy;

    MPI_Datatype mpi_datatype = std::is_same<FloatingType, double>::value ? MPI_C_DOUBLE_COMPLEX : MPI_C_FLOAT_COMPLEX;

    // Retrieve the matrix dimensions
    const auto& dims = global_tensor.dimensions();
    
    // Rows
    CTensor_1D axis_tensor;
    for(int x=0; x<dims[0]; x++){
        // Curent 
        axis_tensor = global_tensor.chip(x,0);
        strategy.fft(axis_tensor, fftDirection);   
        global_tensor.chip(x,0) = axis_tensor;
    }
     
    MPI_Bcast(global_tensor.data(), global_tensor.size(), mpi_datatype, 0, MPI_COMM_WORLD);

    // Columns 
    for(int y=0; y<dims[1]; y++){
        axis_tensor= global_tensor.chip(y,1);
        strategy.fft(axis_tensor, fftDirection);    
        global_tensor.chip(y,1) = axis_tensor;
    }
    
    MPI_Bcast(global_tensor.data(), global_tensor.size(), mpi_datatype, 0, MPI_COMM_WORLD);

}


#endif 
