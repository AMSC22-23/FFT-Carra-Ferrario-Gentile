#include <string>
#include "TensorFFTBase.hpp"
/**
 * This template extends the TensorFFTBase template to
 * handle data in the time domain, providing additional
 * utility methods.
 * @author: Daniele Ferrario
*/
template<typename DataType, int Rank>
class TimeTensor : public TensorFFTBase<DataType, Rank>{
    public:

        // Inherit constructor
        using TensorFFTBase<DataType, Rank>::TensorFFTBase;

        // @Todo: data splicing methods
        void load_from_file(std::string path){
            // @Todo
        }

};
