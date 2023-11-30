#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include <sys/time.h>
#include <iostream>

int main(void){

    int N = 10000;
    Eigen::MatrixXd EigenMatrix(N, N);
    Eigen::Tensor<double, 2> EigenTensor(N, N);

    EigenMatrix = Eigen::MatrixXd::Random(N, N);

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            EigenMatrix(i, j) = EigenMatrix(j, i);
        }
    }

    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << "tensor" << std::endl;
    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n"; 


    t1 = high_resolution_clock::now();
    
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            EigenTensor(i, j) = EigenTensor(i, j);
        }
    }

    t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    ms_double = t2 - t1;

    std::cout << "matrix" << std::endl;
    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";

    return 0;
    
}