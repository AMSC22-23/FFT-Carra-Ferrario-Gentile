#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include <sys/time.h>
#include <iostream>
#include <chrono>

int main(void){

    int N = 20000;
    Eigen::MatrixXd EigenMatrix(N, N);
    Eigen::Tensor<double, 2> EigenTensor(N, N);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            EigenMatrix(i, j) = i * j;
            EigenTensor(i, j) = i * j;
        }
    }

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            EigenMatrix(i, j) = EigenMatrix(j, i) * i * j;
        }
    }

    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << "matrix" << std::endl;
    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n"; 


    t1 = high_resolution_clock::now();
    
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            EigenTensor(i, j) = EigenTensor(j, i) * i * j;
        }
    }

    t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    ms_double = t2 - t1;

    std::cout << "tensor" << std::endl;
    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";

    //print the first 10 elements of the matrix and the tensor

    std::cout << "first 10 elements of the matrix" << std::endl;
    for(int i = 0; i < 10; i++){
        std::cout << EigenMatrix(i, 1) << std::endl;
    }

    std::cout << "first 10 elements of the tensor" << std::endl;
    for(int i = 0; i < 10; i++){
        std::cout << EigenTensor(i, 1) << std::endl;
    }

    return 0;
    
}