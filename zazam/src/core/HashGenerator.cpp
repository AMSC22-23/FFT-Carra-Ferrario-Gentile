#include "HashGenerator.hpp"

using namespace zazamcore; 

void HashGenerator::generate(Matrix_d& spectrogram, Vector_ui& result, bool transpose_spectrogram) {

    assert(spectrogram.size() > 0);

    Matrix_d key_points_matrix;
    
    e_index rows_number = transpose_spectrogram ? spectrogram.dimension(1) : spectrogram.dimension(0);

    e_index cols_number = zazamcore::KeyPointsNumber;

    map_to_key_points_matrix(spectrogram, key_points_matrix, transpose_spectrogram);
    result = Vector_ui(rows_number);

    int hash;
    for(e_index i = 0; i<rows_number; i++){
        hash = cols_number; // Any other number other than 0 will work
        for(e_index j=0; j<cols_number; j++){
            hash = 17 * hash + key_points_matrix(i, j);
        }
        result(i) = hash;
        
    }
        
    assert(result.size() > 0);
}

void HashGenerator::map_to_key_points_matrix(Matrix_d& spectrogram, Matrix_d& key_points_matrix, bool transpose=false) {
    Vector_d row;

    e_index rows_number = transpose ? spectrogram.dimension(1) : spectrogram.dimension(0);

    key_points_matrix = Matrix_d(rows_number, zazamcore::KeyPointsNumber);
    for(e_index y=0; y < rows_number; y++ ){

        if(transpose)
            map_to_key_points_vector(spectrogram.chip(y,1), row);
        else
            map_to_key_points_vector(spectrogram.chip(y,0), row);
            
        key_points_matrix.chip(y,0) = row;

    }

}

void HashGenerator::map_to_key_points_vector(const Vector_d& input, Vector_d& output){
    e_index range_dim =
        (zazamcore::KeyPointsRange.second - zazamcore::KeyPointsRange.first) / zazamcore::KeyPointsNumber;

    e_index offset = zazamcore::KeyPointsRange.first-1;
    Vector_d sub_vector(range_dim);

    output = Vector_d(zazamcore::KeyPointsNumber);
    for(e_index j=0; j<zazamcore::KeyPointsNumber; j++){
        
        // Create subvector
        for(e_index i=0; i<range_dim; i++){
           // std::cout << offset+i << std::endl;
            sub_vector(i) = input(offset+i);
        }

        // Finding the index of the maximum of the sub-vector
        output(j) = offset + utils::find_max_element_index(sub_vector); 

        offset += range_dim; 
    } 
}

