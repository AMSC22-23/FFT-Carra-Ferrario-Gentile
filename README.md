# FFFT - Fast and FOUrious FOUrier Transform 
### A parallel library for the Fast Fourier Transform
AMSC project powered by Carrà Edoardo, Gentile Lorenzo, Ferrario Daniele.

## Introduction
Fast and FOUrious FOUrier Transform (FFFT) is a parallel C++ library designed for computing the Fast Fourier Transform design to be highly extensible, and easy to integrate in different applications.

## Required software

This library depends on two modules: `Eigen v3.3.9` and `fftw3`, both of these modules are available in the **mk modules**.


fftw3 is a state-of-the-art library widely used in the scientific community for its efficiency and accuracy. In the context of this library, it is utilized primarily for testing the correctness and performance of the different implementations.

## Contents
This repository contains four main components:
1. **ffft**: The library itself. [go](./ffft/)
2. **Tests**: This directory contains various test cases for evaluating the performance and accuracy of different implementation strategies in the library. [go](./test/)
3. **Cluster Setup Procedure**: a procedure for setting up a cluster for running MPI tests. [go](./MPI_Cluster_Setup)
4. **Benchmark Tool**: some scripts to run tests on different workloads, in order to compare different FFT strategies. [go](./benchmark/)
5. **Spectogram**: a simple application that computes the audio spectrogram using the ffft library, simulating a real case scenario. [go](./)

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
    FFTSolver<2, double> solver(std::make_unique<SequentialFFT_2D>());

    // Create and initialize the data
    CTensorBase<2, double> data(3,4); 
    data.get_tensor().setRandom();
    
    // Compute the FFT
    solver.compute_fft(data, FFT_FORWARD);
}

```


### Library folder structure
A brief description of the library structure. For more details, always refer to the [documentation](./doc/).
- `ffft/src/fftcore/`: 
    - [`FFTSolver.hpp`](./ffft/src/fftcore/FFTSolver.hpp): this header file provides direct access for users to perform FFT operations.
    - [`Strategy/`](./ffft/src/fftcore/Strategy): this directory contains different strategies for performing the FFT, both in parallel and in sequential. At the moment only 1, 2 and 3 dimensional data are supported.
    - [`Tensor/`](./ffft/src/fftcore/Tensor): a module that contains the main tools to manipulate Eigen's tensor and do pre-processing and post-processing.
    - [`Timer/`](./ffft/src/fftcore/Timer): a module that allows to time the solve method.
    - [`utils/`](./ffft/src/fftcore/utils): a module that contains some library utilities.

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
or if you are not using mk modules, specify fftw installation path insted of *mkFftwPrefix*.

Finally, build the test by typing `cmake --build . --target TESTNAME.out`, choosing from the following list of tests currently available:
```
test_MPI.out
test_MPI_2D.out
test_MPI_3D.out
test_OMP.out
test_OMP_2D.out
test_OMP_3D.out
test_sequential.out
test_sequential_2D.out
test_sequential_3D.out
test_stockham.out
```


An executable for each test will be created into `/build`, and can be executed through
```bash
$ ./test_NAME.out N (M Z)
```
where N is the log2 of the size of the input and M and Z have to be specified in the case of multidimensional tests.

### Documentation
For a comprehensive understanding of the library's structure and usage within your project, please refer to the [documentation](./doc/). This contains a report on the numerical methodology and on the code organization as well as some results on the performance of the library.

