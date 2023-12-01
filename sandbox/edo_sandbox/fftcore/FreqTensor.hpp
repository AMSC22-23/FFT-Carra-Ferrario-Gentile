#include <string>
#include "TensorFFTBase.hpp"

/**
 * This template extends the TensorFFTBase template to
 * handle data in the frequency domain, providing additional
 * utility methods.
 * @author: Daniele Ferrario

*/
template<typename DataType, int Rank>
class FreqTensor : public TensorFFTBase<DataType, Rank>{
    public:
        // Inherit constructor
        using TensorFFTBase<DataType, Rank>::TensorFFTBase;

        void apply_high_pass_filter();
        void apply_low_pass_filter();
        void apply_band_pass_filter();

        void load_from_file(std::string path){
            // @Todo
        }

};

/**
 * @Todo: to implement soon
*/
template<typename DataType, int Rank>
void FreqTensor<DataType, Rank>::apply_high_pass_filter(){

};

template<typename DataType, int Rank>
void FreqTensor<DataType, Rank>::apply_low_pass_filter(){

};

template<typename DataType, int Rank>
void FreqTensor<DataType, Rank>::apply_band_pass_filter(){

};

/** ----------------------*/