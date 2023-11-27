#pragma once

#include <string>
#include <iostream>
#include "TensorFFTBase.hpp"
#include "utils/MtxFilesIO.hpp"


/**
 * This template extends the TensorFFTBase template to
 * handle data in the time domain, providing additional
 * utility methods.
 * @author: Daniele Ferrario
*/
template<typename DataType>
class TimeTensor : public TensorFFTBase<DataType, 1>{
    public:

        // Inherit constructor
        using TensorFFTBase<DataType, 1>::TensorFFTBase;

        // @Todo: time splicing methods
        
        void load_from_file(const std::string&);

};

template<typename DataType>
void TimeTensor<DataType>::load_from_file(const std::string &path){
    MtxFilesIO::load_mat_mtx<DataType, 1>(this->get_tensor(), path);
}