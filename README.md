# FFFT - Fast and FOUrious FOUrier Transform 
### A parallel library for the Fast Fourier Transform
AMSC hands-on powered by Carrà Edoardo, Gentile Lorenzo, Ferrario Daniele.

### Introduction
FFTCore is a parallel C++ library designed for computing the Fast Fourier Transform design to be highly extensible, and easy to integrate in different applications.

### Required software

This library depends on two modules: `Eigen v3.3.9` and `fftw3`, both of these modules are available in the mk modules.


fftw3 is a state-of-the-art library widely used in the scientific community for its efficiency and accuracy. In the context of this library, it is utilized primarily for testing the correctness and performance of the different implementations.

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
and build the test by typing `cmake --build . --target TESTNAME.out`, choosing from the following list:
```
test_MPI.out
test_MPI_2D.out
test_OMP.out
test_OMP_2D.out
test_sequential.out
test_sequential_2D.out
test_stockham.out
```


An executable for each test will be created into `/build`, and can be executed through
```bash
$ ./test_NAME.out
```
### Documentation
For a comprehensive understanding of the library's structure and usage within your project, please refer to the [documentation](./doc/).

