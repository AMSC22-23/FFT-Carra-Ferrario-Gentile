#ifndef MPIFFT_HPP
#define MPIFFT_HPP

#define dout std::cout << "p" << rank <<": "
#define INIT_TIME_MEASURE double start, end;

#define MEASURE_TIME_START start = MPI_Wtime();

#define MEASURE_TIME_END(message)                                                                                            \
    do                                                                                                                       \
    {                                                                                                                        \
        end = MPI_Wtime();                                                                                                   \
        if (rank == 0)                                                                                                       \
        {                                                                                                                    \
            /*std::cout << "p: " << rank << " " << message << ": " << (end - start)*1.0e6 << " microseconds" << std::endl;*/ \
        }                                                                                                                    \
    } while (0);


#include <iostream>
#include <mpi.h>
#include "../../../FFTSolver.hpp"
#include "../../../utils/FFTUtils.hpp"

using namespace fftcore;
using namespace std;

template <typename FloatingType = double>
class MPIFFT : public FFT_1D<FloatingType>
{
public:
    using typename FFT_1D<FloatingType>::RTensor_1D;
    using typename FFT_1D<FloatingType>::CTensor_1D;

    void fft(const CTensor_1D &, CTensor_1D &, FFTDirection) const;

    void fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const;

    void fft(CTensor_1D &, FFTDirection) const;

    ~MPIFFT() = default;

private:
    void _generalized_cooleytukey_butterfly(CTensor_1D &, const int) const;
};

template <typename FloatingType>
void MPIFFT<FloatingType>::fft(const CTensor_1D &input, CTensor_1D &output, FFTDirection) const
{
    std::cout << "fft 1-d C-C out-of-place" << std::endl;
};

template <typename FloatingType>
void MPIFFT<FloatingType>::fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const
{
    std::cout << "fft 1-d R-C out-of-place" << std::endl;
};

/**
 * In place FFT using MPI parallelization. Every process will compute a
 * sub tree of the iterational Cooley-tuekey algorithm, until a dependency between them
 * will occure in the butterfly step. Then, only one process will complete the
 * remaining steps of the algorithm, precisely doing log(p_size) steps with p_size
 * being the number of available processes.

 * Remarks:
 *
 * In this implementation of the iterative algorithm, there are two main phases:
 *
 * P1.) The bit reversal permutation of the vector elements
 * P2.) The "butterfly" operations

 * MPI is thought to express coarser parallelism, distributing workload on
 * multiple CPUs, by adopting the Message passing memory model. 
 * 
 * This makes it a good choice for parallelizing tasks on clusters or distributed memory
 * systems where the data is too large to fit into a single node's memory.
 * 
 * Parallelizing P2 is easy, because the processes will work indipendently on different chunks
 * of the input.
 * 
 * Parallelizing P1, with this model, leads to a tradeoff between space complexity and time complexity.
 * 
 * 1.) The first approach would be equally dividing the input on different chunks for each process and run
 *     several message passings between processes to retrieve the data to swap, when the process doesn't own it.
 * 2.) Every process would contain a copy of the global data, and there would be no need to comunicate. 
 * 
 * The second approach has been implemented, being the library time-optimization oriented.
 * 
 * @param The 1-D vector to transorm, already loaded by each of the processes.
 * @author: Daniele Ferrario
 * @TODO: add exceptions instead of assertions
*/

template <typename FloatingType>
void MPIFFT<FloatingType>::fft(CTensor_1D &global_tensor, fftcore::FFTDirection fftDirection) const
{

    // Utility
    using Complex = std::complex<FloatingType>;

    // MPI Infos
    int rank, size;
    MPI_Datatype mpi_datatype = std::is_same<FloatingType, double>::value ? MPI_C_DOUBLE_COMPLEX : MPI_C_FLOAT_COMPLEX;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Tensor infos
    int n = global_tensor.size();
    int log2n = std::log2(n);
    double log2p = std::log2(size);

    // Assertions
    assert(!(n & (n - 1)) && "FFT length must be a power of 2.");
    assert((size <= n / 2 && std::ceil(log2p) == std::floor(log2p)) && "Process number must be a power of two and less or equal than n/2).");
    // -------------------------------------

    // Length of the sub-vector related to each process
    int local_tensor_size = global_tensor.size() / size;
    // Global index of the sub-vector
    int offset = rank * local_tensor_size;

    // Every process runs a bit reversal permutation towards
    // a sub-vector, still by working on its copy of the global
    // vector.

    int rev;
    for (int i = 0; i < local_tensor_size; i++)
    {
        rev = FFTUtils::reverseBits(offset + i, log2n);

        // Avoid to swap elements if the reverse
        // has already been processed by the process ( rev > (offset+i) ),
        // or if it's placed in a sub vector
        // of another process competence ( rev < offset )

        if (rev > (offset + i) || rev < offset)
        {

            // If the elements to swap are in the same range
            // of the process, swap them.

            if ((rev / local_tensor_size) == rank)
            {
                std::swap(global_tensor[offset + i], global_tensor[rev]);
            }
            else
            {

                // Else, there is no need to copy the current position
                // in the reversed position. This is because it will be
                // discarded at MPI_Gather as it's not this processor competence
                // to handle it.

                global_tensor[offset + i] = global_tensor[rev];
            }
        }
    }

    // Gather the permuted sub-vectors of each processor competence into process 0.
    MPI_Gather(global_tensor.data() + offset, local_tensor_size, mpi_datatype, global_tensor.data(), local_tensor_size, mpi_datatype, 0, MPI_COMM_WORLD);

    // Conjugate
    if (rank == 0)
    {
        if (fftDirection == fftcore::FFT_INVERSE)
        {
            FFTUtils::conjugate(global_tensor);
        }
    }

    // Local section of the vector
    CTensor_1D local_tensor(local_tensor_size);




    // Scatter the permutated (and conjugated) global tensor on all processes.
    MPI_Scatter(global_tensor.data(), local_tensor_size, mpi_datatype, local_tensor.data(), local_tensor_size, mpi_datatype, 0, MPI_COMM_WORLD);

    // Run fft on local tensors ( subtrees )
    MPIFFT::_generalized_cooleytukey_butterfly(local_tensor, 1);

    // Reconstruct the partially computed vector
    MPI_Gather(local_tensor.data(), local_tensor_size, mpi_datatype, global_tensor.data(), local_tensor_size, mpi_datatype, 0, MPI_COMM_WORLD);

    // Sequentially compute the remaining steps
    if (rank == 0)
    {

        // Now there remains log2(num of processes) steps to do
        // If there is only one process, butterfly has already been completed
        if (log2p > 0)
            MPIFFT::_generalized_cooleytukey_butterfly(global_tensor, log2n - log2p + 1);

        // Re-conjugate and scale if inverse
        if (fftDirection == fftcore::FFT_INVERSE)
        {
            FFTUtils::conjugate(global_tensor);
            global_tensor = global_tensor * Complex(1.0 / n, 0);
        }
    }

    // The global vector is ready on process 0.
}

/**
 * This private method is employed when dealing with the butterfly phase
 * of cooley-tukey algoirithm
 * @param input_output The local vector
 * @param starting_depth The current depth (1...log(range)) to start from, since the input
 * could have already been manipulated previously.
 */
template <typename FloatingType>
void MPIFFT<FloatingType>::_generalized_cooleytukey_butterfly(CTensor_1D &input_output, const int starting_depth) const
{
    using Complex = std::complex<FloatingType>;
    int n = input_output.size();
    assert(!(n & (n - 1)) && "FFT length must be a power of 2.");
    int log2n = std::log2(n);
    assert((starting_depth >= 1 && starting_depth <= log2n) && "Starting depth has to be in (1...log2(n)).");

    Complex w, wm, t, u;
    int m, m2;
    // Cooley-Tukey iterative FFT
    for (int s = starting_depth; s <= log2n; ++s)
    {
        m = 1 << s;                          // 2 power s
        m2 = m >> 1;                         // m2 = m/2 -1
        wm = exp(Complex(0, -2 * M_PI / m)); // w_m = e^(-2*pi/m)

        for (int k = 0; k < n; k += m)
        {
            w = Complex(1, 0);
            for (int j = 0; j < m2; ++j)
            {
                t = w * input_output[k + j + m2];
                u = input_output[k + j];

                input_output[k + j] = u + t;
                input_output[k + j + m2] = u - t;

                w *= wm;
            }
        }
    }
};

#endif // MPIFFT_HPP
