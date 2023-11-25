
#include <string>
#include <iostream>
#include "TensorFFTBase.hpp"
#include "utils/EigenTensorFilesIO.hpp"


/**
 * This template extends the TensorFFTBase template to
 * handle images data (Rank 2 tensors), providing additional
 * utility methods.
 * @TODO: How images are really stored in a rank 2 tensor?
 * @author: Daniele Ferrario
*/
template<typename DataType>
class ImageTensor : public TensorFFTBase<DataType, 2>{
    public:

        // Inherit constructor
        using TensorFFTBase<DataType, 2>::TensorFFTBase;

        // @Todo: data splicing methods
        void load_from_file(std::string);

};

template<typename DataType>
void ImageTensor<DataType>::load_from_file(std::string path){
    EigenTensorFilesIO::load_2d_mtx<DataType>(this->get_tensor(), path);
}