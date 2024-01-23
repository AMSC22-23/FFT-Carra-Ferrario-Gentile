
#ifndef MPI_OMP_FFT_2D_HPP
#define MPI_OMP_FFT_2D_HPP


#include <iostream>
#include <mpi.h>
#include "../../../FFTSolver.hpp"
#include "../../../utils/FFTUtils.hpp"

using namespace std;

namespace fftcore{

    /**
     * @brief hybrid MPI and OpenMP implementation of the 2 Dimensional FFT 
     * 
     * @author Edoardo Carrà
    */
    template <typename FloatingType = double>
    class MPI_OMP_FFT_2D : public FFT_2D<FloatingType>
    {
    public:

        // @TODO: Find a way to import from FFT_1D 
        using typename FFT_2D<FloatingType>::CTensor_1D;
        using typename FFT_2D<FloatingType>::RTensor_1D;

        using typename FFT_2D<FloatingType>::RTensor_2D;
        using typename FFT_2D<FloatingType>::CTensor_2D;
        
        void fft(const CTensor_2D &, CTensor_2D &, FFTDirection) const;

        void fft(const RTensor_2D &, CTensor_2D &, FFTDirection) const;

        void fft(CTensor_2D &, FFTDirection) const;

        ~MPI_OMP_FFT_2D() = default;

    };

    /**
     * @brief Out-of-place of 2-D complex tensors (Matrices)
    */
    template <typename FloatingType>
    void MPI_OMP_FFT_2D<FloatingType>::fft(const CTensor_2D &input, CTensor_2D &output, FFTDirection) const
    {
        std::cout << "fft 2-d C-C out-of-place" << std::endl;
    };

    /**
     * @brief Out-of-place Real to Complex of real 2-D tensors (Matrices)
    */
    template <typename FloatingType>
    void MPI_OMP_FFT_2D<FloatingType>::fft(const RTensor_2D &, CTensor_2D &, FFTDirection) const
    {
        std::cout << "fft 2-d R-C out-of-place" << std::endl;
    };

    /**
     * The method implements an hybryd MPI+OMP implementation of the 2 dimensional FFT. The transformation begins
     *  with the application of the Cooley-Tukey algorithm to the columns of the two-dimensional data (FIRST PHASE). 
     * Following this, the same algorithm is applied to the rows (SECOND PHASE).
     * The access pattern adopted in this method is characterized by an extended stencil. This means that the
     * transformation within any given cell of the tensor relies on every element in the same row and column.
     * We can generalize this by dividing the matrix into multiple blocks that follows the same dependency pattern.
     * Each block has dimension x_dimension_size/size X y_dimension_size/size, where the original tensor has dimensions
     * x_dimension_size X y_dimension_size and size is the number of MPI process. Since at least one axis of the 
     * original tensor needs to be aligned (non-distributed), one process owns only the blocks on the same row or on the same 
     * column.
     * So this means that after the first phase (transformation of the columns) the process should exchange the result of the first
     * computation in order to compute the transform along the other direction.
     * 
     * The method performs the following steps:
     * - At the beginning, the input tensor is owned only by the root process. 
     * - **FIRST PHASE** :
     * 
     *          1) Distribution: the tensor is distributed with MPI_Scatter along the x-direction following the
     *             col-major format of Eigen's tensors. Each process saves y_dimension_size/size columns of the original 
     *             tensor in a local tensor. The block distribution of the 2D global tensor is saved in block_distribution, 
     *             a size x size matrix, where each element contains the block owner's rank.
     *             
     *             An example of distribution of a 8x16 2-D tensor (row,col):
     *             +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+------+------+------+------+------+------+
     *             | 0,0 | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 | 0,6 | 0,7 | 0,8 | 0,9 | 0,10 | 0,11 | 0,12 | 0,13 | 0,14 | 0,15 |
     *             | 1,0 | 1,1 | 1,2 | 1,3 | 1,4 | 1,5 | 1,6 | 1,7 | 1,8 | 1,9 | 1,10 | 1,11 | 1,12 | 1,13 | 1,14 | 1,15 |
     *             | 2,0 | 2,1 | 2,2 | 2,3 | 2,4 | 2,5 | 2,6 | 2,7 | 2,8 | 2,9 | 2,10 | 2,11 | 2,12 | 2,13 | 2,14 | 2,15 |
     *             | 3,0 | 3,1 | 3,2 | 3,3 | 3,4 | 3,5 | 3,6 | 3,7 | 3,8 | 3,9 | 3,10 | 3,11 | 3,12 | 3,13 | 3,14 | 3,15 |
     *             | 4,0 | 4,1 | 4,2 | 4,3 | 4,4 | 4,5 | 4,6 | 4,7 | 4,8 | 4,9 | 4,10 | 4,11 | 4,12 | 4,13 | 4,14 | 4,15 |
     *             | 5,0 | 5,1 | 5,2 | 5,3 | 5,4 | 5,5 | 5,6 | 5,7 | 5,8 | 5,9 | 5,10 | 5,11 | 5,12 | 5,13 | 5,14 | 5,15 |
     *             | 6,0 | 6,1 | 6,2 | 6,3 | 6,4 | 6,5 | 6,6 | 6,7 | 6,8 | 6,9 | 6,10 | 6,11 | 6,12 | 6,13 | 6,14 | 6,15 |
     *             | 7,0 | 7,1 | 7,2 | 7,3 | 7,4 | 7,5 | 7,6 | 7,7 | 7,8 | 7,9 | 7,10 | 7,11 | 7,12 | 7,13 | 7,14 | 7,15 |
     *             +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+------+------+------+------+------+------+
     *           
     *             with rank=4, each block has size 2x4 and block_disposition has the following structure:
     *             +---+---+---+---+
     *             | 0 | 1 | 2 | 3 |
     *             | 0 | 1 | 2 | 3 |
     *             | 0 | 1 | 2 | 3 |
     *             | 0 | 1 | 2 | 3 |
     *             +---+---+---+---+
     *              
     *          2) Transformation: using OmpFFT strategy for computing 1 dimensional data, each column is transformed using 
     *             Cooley-Tukey algorithm.
     * 
     *              In the previous example, each MPI process will compute four 1 dimensional fft using omp.
     * 
     * - SECOND PHASE :
     * 
     *          1) Distribution: the distribution in this case is less trivial. Each process should send the first block to the
     *             rank 0 process, the second to the rank 1 process and so on... This can be achieved with MPI_Alltoallv, that
     *             allows to create a point-to-point communication for each pair of process. The new block distribution can be 
     *             obtained by transposing the block_distribution.
     *              
     *             Following the example, the block distribution of the global vector in the second phase is the following:
     *             +---+---+---+---+
     *             | 0 | 0 | 0 | 0 |
     *             | 1 | 1 | 1 | 1 |
     *             | 2 | 2 | 2 | 2 |
     *             | 3 | 3 | 3 | 3 |
     *             +---+---+---+---+
     *             
     *              
     *          2) Transformation: using OmpFFT strategy for computing 1 dimensional data, each row is transformed using 
     *             Cooley-Tukey algorithm.
     * 
     * - FINAL PHASE : each row is gathered by the rank 0 process and saved the global tensor. 
     * 
     * 
     * @author Edoardo Carrà
    */
    template <typename FloatingType>
    void MPI_OMP_FFT_2D<FloatingType>::fft(CTensor_2D &global_tensor, FFTDirection fftDirection) const
    {

        // initialize mpi data
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Datatype mpi_datatype = std::is_same<FloatingType, double>::value ? MPI_C_DOUBLE_COMPLEX : MPI_C_FLOAT_COMPLEX;

        assert(!(size & (size - 1)) && "MPI SIZE must be a power of 2.");

        unsigned int x_dimension_size, y_dimension_size;
        void* tensor_data;
        
        OmpFFT omp_1D_solver;
        CTensor_1D axis_input;

        if(rank==0){
            const auto& dimensions = global_tensor.dimensions();
            x_dimension_size = dimensions[0];
            y_dimension_size = dimensions[1];
        
            //input dimensions check
            for(int i=0; i<2; i++){
                assert(!(dimensions[i] & (dimensions[i] - 1)) && "All dimensions of the input must be a power of 2.");
            }

            tensor_data = global_tensor.data();
        }

        //##########################################################################
        //                             first phase
        //##########################################################################
        
        //--------------------------------------------------------------------------
        //                             distribution
        //--------------------------------------------------------------------------

        //broadcast x and y tensor's dimension
        MPI_Bcast(&x_dimension_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        MPI_Bcast(&y_dimension_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        //calculate block size
        const unsigned int x_block_size = x_dimension_size/size;
        const unsigned int y_block_size = y_dimension_size/size;
        
        // Create the block disposition of the first phase of the transformation.
        // Each element of block_distribution represents the block owner's rank.
        Eigen::MatrixXi block_distribution;
        block_distribution.resize(size, size);

        // col-major displacement of blocks
        for(int i=0; i<size; i++){
            for(int j=0; j<size; j++){
                block_distribution(i,j)=j;
            }
        }

        // dimension of the local tensor
        const unsigned int x_local_dimension_size=x_dimension_size;
        const unsigned int y_local_dimension_size=y_dimension_size/size;
        const unsigned int local_tensor_elements_number = x_local_dimension_size*y_local_dimension_size;
            
        //local tensor
        CTensor_2D local_tensor(x_local_dimension_size, y_local_dimension_size);
        void* local_tensor_data = local_tensor.data();

        MPI_Scatter(tensor_data, local_tensor_elements_number, mpi_datatype, local_tensor_data, local_tensor_elements_number, mpi_datatype,0, MPI_COMM_WORLD);
        

        //--------------------------------------------------------------------------
        //                             transformation
        //--------------------------------------------------------------------------
        for(Eigen::Index line=0; line< y_local_dimension_size; line++){
            axis_input = local_tensor.chip(line,1);
            omp_1D_solver.fft(axis_input, fftDirection);
            local_tensor.chip(line,1) = axis_input;
        }
        

        //##########################################################################
        //                             second phase
        //##########################################################################
        
        //--------------------------------------------------------------------------
        //                             distribution
        //--------------------------------------------------------------------------

        // Create the block distribution in the second phase of the transformation 
        // transposing block_distribution
        const Eigen::MatrixXi block_distribution_2phase = block_distribution.transpose();

        /*
         * Preparing MPI_Alltoallv data structures
         */

        // sending 
        std::vector<int> send_counts(size, x_block_size);
        // Is an integer vector where entry i specifies the displacement (offset from sendbuf, in units of sendtype) 
        // from which to send data to rank i.
        std::vector<int> send_displ(size);

        for(int i=0; i<size; i++){
            // get where to send each block thanks to the block_distribution_2phase
            send_displ[i] = block_distribution_2phase(i,rank)*x_block_size;
        }
        
        // for all columns of the local_tensor we have to perform an MPI_Alltoallv due to col-major format
        for(int local_col_index=0; local_col_index<y_local_dimension_size;local_col_index++){
            int starting_index_of_the_local_col = local_col_index*x_dimension_size;
            MPI_Alltoallv(MPI_IN_PLACE, NULL, NULL, MPI_DATATYPE_NULL,
                        &local_tensor.data()[starting_index_of_the_local_col], send_counts.data(), send_displ.data(), mpi_datatype, MPI_COMM_WORLD);
        }

        CTensor_2D local_tensor_transposed(x_block_size, y_block_size*size);
        const unsigned int x_local_transposed_dimension_size = x_block_size;
        const unsigned int y_local_transposed_dimension_size = y_block_size*size;
        
        // block transposition in order to use the 
        for(int n_block=0; n_block<size; n_block++)
        {
            for(int i=0; i<x_block_size; i++){
                for(int j=0; j<y_block_size; j++){
                    local_tensor_transposed(i,j+n_block*y_block_size) = local_tensor(i+n_block*x_block_size,j);
                }
            }
        }

        
        //--------------------------------------------------------------------------
        //                             transformation
        //--------------------------------------------------------------------------
        for(Eigen::Index line=0; line< x_local_transposed_dimension_size; line++){
            axis_input = local_tensor_transposed.chip(line,0);
            omp_1D_solver.fft(axis_input, fftDirection);
            local_tensor_transposed.chip(line,0) = axis_input;
        }

        
        //##########################################################################
        //                             final phase
        //##########################################################################

        //gather all rows
        int col_local, col_global;
        for(int j=0; j<y_dimension_size; j++){
            col_local = j*x_block_size;
            col_global = j*x_dimension_size;

            MPI_Gather(&local_tensor_transposed.data()[col_local], x_block_size, mpi_datatype,
                    &global_tensor.data()[col_global], x_block_size, mpi_datatype, 0, MPI_COMM_WORLD );
        }

    }
}


#endif 
