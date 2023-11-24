#include "FFTDataTypes.hpp"
#include <unsupported/Eigen/CXX11/Tensor>



int main(){
    int n = 3;
    FFTDataTypes::TimeTensor<int, 3> time_tensor(n,n,n);
    FFTDataTypes::FreqTensor<int, 3> freq_tensor(n,n,n);

    int k = 0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            time_tensor(i, j, 1) = k++;
        }
    }
    std::cout << time_tensor << std::endl;
    
    return 0;
}