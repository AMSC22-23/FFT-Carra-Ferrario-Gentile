# FFFT - Fast and FOUrious FOUrier Transform 
### A parallel library for the Fast Fourier Transform
AMSC project powered by Carrà Edoardo, Gentile Lorenzo, Ferrario Daniele.

## Introduction
FFTCore is a parallel C++ library designed for computing the Fast Fourier Transform design to be highly extensible, and easy to integrate in different applications.

## Required software

This library depends on two modules: `Eigen v3.3.9` and `fftw3`, both of these modules are available in the mk modules.


fftw3 is a state-of-the-art library widely used in the scientific community for its efficiency and accuracy. In the context of this library, it is utilized primarily for testing the correctness and performance of the different implementations.

## Contents
This repository contains four main components:
1. **ffft**: The library itself.
2. **Tests**: This directory contains various test cases for evaluating the performance and accuracy of different implementation strategies in the library.
3. **Cluster Setup Procedure**: a procedure for setting up a cluster for running MPI tests. 
4. **Benchmark Tool**: some scripts to run tests on different workloads, in order to compare different FFT strategies.
5. **Spectogram**: a simple application that computes the audio spectrogram using the ffft library, simulating a real case scenario.

## The library
### A simple example
The library is designed to be fast and easy to use. Here is a simple example of how to compute the FFT on a two-dimensional object:

1. Include the library:
```c++
#include "ffft.hpp"
```

2. Create a solver and assign it to a strategy, which in this case is the Cooley-Tuckey sequential FFT for two dimensional objects working with double: 
```c++
FFTSolver<2, double> solver(std::make_unique<SequentialFFT_2D>());
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
The complete code is the following:
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
- `fftcore\`: this directory contains the core files of the library.
    - `FFTSolver.hpp`: this header file provides direct access for users to perform FFT operations.
    - `Strategy\`: this directory contains different strategies for performing the FFT. Some basic implementations are provided by default as the sequential one, MPI, OMP and CUDA.
    - `Tensor\`: this directory contains the main tools to manipulate Eigen's tensor and do pre-processing and post-processing.



### Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load eigen fftw
```
Then, from the root folder, run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake -DFFTW3_DIR=${mkFftwPrefix} ..
```
and build the test by typing `cmake --build . --target TESTNAME.out`, choosing from the following list of strategies:
```
test_MPI.out
test_MPI_2D.out
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
$ ./test_NAME.out N
```
where N is the log2 of the size of the input.

### Documentation
For a comprehensive understanding of the library's structure and usage within your project, please refer to the [documentation](./doc/).

