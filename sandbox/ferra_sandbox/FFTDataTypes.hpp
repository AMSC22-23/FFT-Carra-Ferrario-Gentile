#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

namespace FFTDataTypes{
    template<typename DataType, size_t Rank>
    using TimeTensor = Eigen::Tensor<DataType, Rank>;
    template<typename DataType, size_t Rank>
    using FreqTensor = Eigen::Tensor<DataType, Rank>;
}