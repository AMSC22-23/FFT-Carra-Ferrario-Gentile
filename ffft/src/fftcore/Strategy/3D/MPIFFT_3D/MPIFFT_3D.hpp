
#ifndef MPIFFT_3D_HPP
#define MPIFFT_3D_HPP

#include "../../../FFTSolver.hpp"
#include "../../../utils/FFTUtils.hpp"
#include "../../FFTStrategy.hpp"
#include "../../2D/MPIFFT/MPIFFT_2D.hpp"
#include <iostream>
#include <mpi.h>

using namespace fftcore;
using namespace std;

/**
 * @brief MPIFFT_3D strategy simply executes the 2D fft 
 * on every slice of the input tensor.
 * 
 * @author Daniele Ferrario 
*/
template <typename FloatingType = double>
class MPIFFT_3D : public FFT_3D<FloatingType>
{
public:
    using typename FFT_3D<FloatingType>::RTensor_3D;
    using typename FFT_3D<FloatingType>::CTensor_3D;

    // @TODO: Find a way to import from FFT_2D 
    using typename FFT_3D<FloatingType>::CTensor_2D;
    using typename FFT_3D<FloatingType>::RTensor_2D;
    
    void fft(const CTensor_3D &, CTensor_3D &, FFTDirection) const;

    void fft(const RTensor_3D &, CTensor_3D &, FFTDirection) const;

    void fft(CTensor_3D &, FFTDirection) const;

    ~MPIFFT_3D() = default;

};

/**
 * @brief Out-of-place of 3-D complex tensors 
*/
template <typename FloatingType>
void MPIFFT_3D<FloatingType>::fft(const CTensor_3D &input, CTensor_3D &output, FFTDirection) const
{
    std::cout << "fft 3-d C-C out-of-place" << std::endl;
};

/**
 * @brief Out-of-place Real to Complex of real 3-D tensors 
*/
template <typename FloatingType>
void MPIFFT_3D<FloatingType>::fft(const RTensor_3D &, CTensor_3D &, FFTDirection) const
{
    std::cout << "fft 3-d R-C out-of-place" << std::endl;
};

/**
 * @brief In-place fft of 3-D complex tensors (Matrices).
 * @see MPIFFT_3D for details. 
 * @param &global_tensor is the 3D Eigen Tensor to transform.
*/
template <typename FloatingType>
void MPIFFT_3D<FloatingType>::fft(CTensor_3D &global_tensor, fftcore::FFTDirection fftDirection) const
{
    
    MPIFFT_2D strategy;

    MPI_Datatype mpi_datatype = std::is_same<FloatingType, double>::value ? MPI_C_DOUBLE_COMPLEX : MPI_C_FLOAT_COMPLEX;

    // Retrieve the matrix dimensions
    const auto& dims = global_tensor.dimensions();
    
    // Along X axis 
    CTensor_2D axis_tensor;
    for(int x=0; x<dims[0]; x++){
        // Curent 
        axis_tensor = global_tensor.chip(x,0);
        strategy.fft(axis_tensor, fftDirection);   
        global_tensor.chip(x,0) = axis_tensor;
    }
     
    MPI_Bcast(global_tensor.data(), global_tensor.size(), mpi_datatype, 0, MPI_COMM_WORLD);

    // Along Y axis 
    for(int y=0; y<dims[1]; y++){
        axis_tensor= global_tensor.chip(y,1);
        strategy.fft(axis_tensor, fftDirection);    
        global_tensor.chip(y,1) = axis_tensor;
    }

    MPI_Bcast(global_tensor.data(), global_tensor.size(), mpi_datatype, 0, MPI_COMM_WORLD);

    // Along Z axis 
    for(int z=0; z<dims[2]; z++){
        axis_tensor= global_tensor.chip(z,2);
        strategy.fft(axis_tensor, fftDirection);    
        global_tensor.chip(z,2) = axis_tensor;
    }

    MPI_Bcast(global_tensor.data(), global_tensor.size(), mpi_datatype, 0, MPI_COMM_WORLD);

}


#endif 
