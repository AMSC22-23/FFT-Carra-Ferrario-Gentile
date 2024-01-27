# FFFT - Fast and FOUrious FOUrier Transform 
### A parallel library for the Fast Fourier Transform
AMSC project powered by Carrà Edoardo, Gentile Lorenzo, Ferrario Daniele.

## Introduction
Fast and FOUrious FOUrier Transform (FFFT) is a parallel C++ library designed for computing the Fast Fourier Transform designed to be highly extensible, and easy to integrate in different applications.

## Required software

This library depends on two modules: `Eigen v3.3.9` and `fftw3`, both of these modules are available in the **[mk modules](https://github.com/pcafrica/mk)**.

fftw3 is a state-of-the-art library widely used in the scientific community for its efficiency and accuracy. In the context of this library, it is utilized primarily for testing the correctness and performance of the different implementations.
### For CUDA
The ffft library provides a CUDA implementation of the FFT. In order to compile the library with CUDA support, you need to have a version of Cuda Toolkit installed on your system. The library has been tested with version 12.3.\
**Important**: CUDA is incompatible with the version of Eigen bundled in the mk modules (Eigen 3.3.9). If you want to use CUDA, you need at least Eigen 3.4.0. You can download it from [here](https://gitlab.com/libeigen/eigen/-/releases/3.4.0). 

## Contents
This repository contains 2 main components:

1. 🛠️ **[ffft](./ffft)**: The library itself. It is made up of 2 modules: the **fftcore** module and the **Spectrogram** module. The former contains a collection of various FFT algorithm implementations, while the latter enables the creation of a visual representation of the spectrum of frequencies in a signal as it varies over time.
2. 🎵 **[Zazam](./zazam)**: a music identification library based on ffft **Spectrogram** module.

And 3 utility components:

5. **[Tests](./test)**: This directory contains various test cases for evaluating the performance and accuracy of different implementation strategies in the library.
6. **[Cluster Setup Procedure](./MPI_Cluster_Setup/)**: a procedure for setting up a cluster for running MPI tests.
7. **[Benchmark Tool](./benchmark/)**: some scripts to run tests on different workloads, in order to compare different FFT strategies.


## The FFFT library
### A simple example
The library is designed to be fast and easy to use. Here is a simple example of how to compute the FFT on a two-dimensional object using the FFFT library:

1. First we include the library:
```c++
#include "ffft.hpp"
```

2. Create a solver and assign it to a strategy, which in this case is the Cooley-Tukey sequential FFT for two dimensional objects working with double: 
```c++
FFTSolver<2, double> solver(std::make_unique<SequentialFFT_2D<double>>());
 ```

3. Create and initialize randomly the structure to store the two-dimensional data of size 3x4, using the default wrapper provided by the library:
```c++
CTensorBase<2, double> data(3,4); 
data.get_tensor().setRandom();
```

4. Finally transform the data using *solve* (in-place):
```c++
solver.compute_fft(data, FFT_FORWARD);
// the results are available in data
```

Complete code:
```c++
#include "ffft.hpp"

int main(){
    // Create the solver
    FFTSolver<2, double> solver(std::make_unique<SequentialFFT_2D<double>>());

    // Create and initialize the data
    CTensorBase<2, double> data(3,4); 
    data.get_tensor().setRandom();
    
    // Compute the FFT
    solver.compute_fft(data, FFT_FORWARD);
}

```


### Library folder structure
A brief description of the library structure. For more details, always refer to the [documentation](./doc/).
- `ffft/src/`:
    - [`fftcore/`](./ffft/src/fftcore/): the core module of the library.
        - [`FFTSolver.hpp`](./ffft/src/fftcore/FFTSolver.hpp): this header file provides direct access for users to perform FFT operations.
        - [`Strategy/`](./ffft/src/fftcore/Strategy): this directory contains different strategies for performing the FFT, both in parallel and in sequential. At the moment only 1, 2 and 3 dimensional data are supported.
        - [`Tensor/`](./ffft/src/fftcore/Tensor): a module that contains the main tools to manipulate Eigen's tensors and do pre-processing and post-processing.
        - [`Timer/`](./ffft/src/fftcore/Timer): a module that allows to time the solve method.
        - [`utils/`](./ffft/src/fftcore/utils): a module that contains some library utilities.
    - [`spectrogram/`](./ffft/src/spectrogram/): a module that provides methods to compute spectrograms.

### Compiling
To build the executable, make sure you have loaded the needed modules (ONLY IF YOU ARE USING mk modules)
```bash
$ module load eigen fftw
```
Then, from the root folder, run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake -DFFTW3_DIR=${mkFftwPrefix} ..
```
or if you are not using mk modules, specify fftw installation path insted of *mkFftwPrefix*.\
Here you can also specify the flag `-DUSE_FAST_MATH=ON` to enable aggressive optimizations for math operations (use with caution).

Now you can build the shared object of the library with the command `make FFFT`, or directly build the tests with `make` (builds all available tests) or `make TESTNAME.out` (builds a specific test). The available tests are:


- test_MPI.out
- test_MPI_2D.out
- test_MPI_3D.out
- test_OMP.out
- test_OMP_2D.out
- test_OMP_3D.out
- test_MPI_OMP_2D.out
- test_sequential.out
- test_sequential_2D.out
- test_sequential_3D.out
- test_stockham.out
- test_spectrogram.out
- test_CUDA.out


An executable for each test will be created into `/build/test`, and can be executed through
```bash
$ ./test/test_NAME.out N (M Z)
```
where N is the log2 of the size of the input and M and Z have to be specified in the case of multidimensional tests.

Each test assesses the performance of a specified strategy against a second one. The evaluation is conducted on a dataset of complex numbers of size $2^n$ (or $\times 2^m \times 2^z$ for multidimensional datasets), populated with random values. The strategies are evaluated in both the forward and inverse directions.

The output of the test is as follows:
`n(,m,z), time_forward_tested_strategy, time_forward_baseline_strategy, time_inverse_tested_strategy, time_inverse_baseline_strategy, speedup_forward, speedup_inverse, error_forward, error_inverse`

### Documentation
For a comprehensive understanding of the library's structure and usage within your project, please refer to the [documentation](./doc/). This contains a report on the numerical methodology and on the code organization as well as some results on the performance of the library.
## Spectrogram application
In the `spectrogram/` directory we provide a small application that uses the Spectrogram module of the library to compute the spectrograms of a set of audio files. To use it please refer to the [spectrogram README](./spectrogram/README.md).
## Zazam
Zazam is a direct example of use of FFFT library. See the details at the corresponding folder.
[Go to Zazam README.md](./zazam/README.md)
