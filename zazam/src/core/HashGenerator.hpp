#ifndef HASHGENERATOR_HPP
#define HASHGENERATOR_HPP

#include "utils/ZazamDataTypes.hpp"
#include "utils/Utils.hpp"

#include <iostream>
namespace zazamcore{
    /**
     * @brief HashGenerator takes care of the generation of the Hash vector related to a certain song or sample.
     * @author Daniele Ferrario 
    */
    class HashGenerator{

        public:
            HashGenerator(){};
            /**
             * @brief Generate the hash from spectrogram data.
             * @param spectrogram_data The spectrogram data: each row representing a time step and each column a frequency. 
             * Please set transpose_spectrogram flag if the roles are inverted. 
             * @param result The resulting hash vector
             * @param transpose_spectrogram Set true if the provided spectrogram data represents a time step at each column and a frequency at each row  
            */
            void generate(Matrix_d &spectrogram_data, Vector_ui &result, bool transpose_spectrogram = false) ;
        private:

            /**
             * @brief Reduce a full spectrogram to a matrix representing the key points. For every row of the matrix, the function map the the initial spectrogram 
             * to a matrix with the frequencies with the max amplitude between certain ranges. ("Song Key Points")
            */
            void map_to_key_points_matrix(Matrix_d &spectrogram, Matrix_d &,bool);
            /**
             * @brief Reduce a vector of the spectrogram to a vector of the matrix representing the key points. 
             *
            */
            void map_to_key_points_vector(const Vector_d& , Vector_d& );
    };
}
#endif