#ifndef MTXFILESIO_HPP
#define MTXFILESIO_HPP

#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <complex>
#include <type_traits>
#include "MtxFilesIOUtils.hpp"


/**
 * IO utils for Eigen vectors, matrixes and tensors.
 * @author Daniele Ferrario
*/
namespace fftcore{
    namespace utils{

        /**
         * Loads a 1-d vector or a 2-d matrix from .mtx file into Eigen::Tensor object.
         * 
         * @TODO: Unfortunally, to load a Dense matrix (and then convert it to a Tensor), 
         * we must pass it through a Sparse Matrix first since loadMarketDense is not present
         * in the current version of Eigen.
         * 
         * @TODO: Maybe use only the load_tensor_method if native Eigen::loadMarket method
         * is computationally equivalent. ( there would be only one method indipendent from the rank )
         * 
         * @param target The Eigen::Tensor object
         * @param path the file path
        */
        template<typename DataType, int Rank>
        void laod_mat_mtx(Eigen::Tensor<DataType, Rank> &target, const std::string &path){
            static_assert(Rank == 1 || Rank == 2, "Rank must be 1 or 2 to use 'load_vec_or_mat_mtx");

            Eigen::SparseMatrix<double> spMat;
            Eigen::MatrixXd denseMat;
            Eigen::loadMarket(spMat, path);
            denseMat = Eigen::MatrixXd(spMat);

            target = Eigen::TensorMap<Eigen::Tensor<double, Rank>>(denseMat.data(), denseMat.rows(), denseMat.cols());

        }

        
        /**
         * Load a generic ranked Eigen::Tensor object from a .mtx file.
         * 
         * @TODO: Check prestations with regards to the native Eigen::loadMarket method for vectors and matrixes.
         * @TODO: Add here to docs the format of the mtx file.
         * @TODO: Implement Complex data loading and symmetric data loading
        */
        template<typename DataType, int Rank>
        void load_tensor_mtx(Eigen::Tensor<DataType, Rank> &tensor, const std::string &path){

            std::ifstream input_file(path);
            std::string line;
            if(!input_file.is_open()){
                std::cerr << "Could not open the file at '" << path << "'." << std::endl;  
            }


            // Data properties
            bool is_complex_data;
            bool is_symmetric; // @TODO: to implement?


            // File line count
            bool reading_tensor_header = true;

            // Non zero elements
            int nnz_header; // Specified in the header
            int nnz_count = 0; // Actual number

            // Size of each dimension
            std::vector<long> dim_sizes;
            
            // Sub string values for each line
            std::vector<std::string> str_values;

            // Current line coordinates
            std::vector<int> current_coordinates;

            // HEADER CHECK    
            std::getline(input_file, line);
            if(!fftcore::utils::load_mtx_tensor_file_settings(line, is_complex_data, is_symmetric)){
                return;
            }

            bool are_complex_data_check = (std::is_same<std::complex<double>, DataType>::value == is_complex_data);
            assert(are_complex_data_check && "Error: file data type doesn't match the provided data structure.");

            // DATA LOADING
            while(std::getline(input_file, line)){
                
                
                str_values = fftcore::utils::split_str_by_whitespace(line);
                
                // Reading tensor specs (second line of file)
                if(reading_tensor_header){

                    // There should be the dimensions for each off the sizes, plus the total elements
                    bool dims_header_check = (str_values.size() == (Rank +1));
                    
                    assert(dims_header_check && "Matrix market header file doesn't match the tensor rank.");

                    
                    // Set the dimension sizes
                    for(int i=0; i<str_values.size()-1; i++){
                        dim_sizes.push_back(std::stol(str_values[i]));
                    }

                    // Set the target nnz entries to load
                    nnz_header = std::stoi(str_values.at(str_values.size()-1));   

                    // Calculating the total tensor size (dim1*dim2*...dimn)
                    int total_tensor_size = 1;
                    for(int size : dim_sizes){
                        total_tensor_size *= size;
                    }
                    
                    // Set the dimensions of the tensor
                    tensor.resize(dim_sizes);

                    tensor.setZero();

                    reading_tensor_header = false;

                }else{
                    
                    // Convert string coordinates to an int vector
                    std::transform(str_values.begin(), str_values.end()-1, std::back_inserter(current_coordinates), [](const std::string& str) {
                        // -1 because of the 1-indexing of mtx file
                        return std::stoi(str)-1;
                    });     


                    // Setting the current line data in the tensor
                    fftcore::utils::tensor_set_value(tensor, current_coordinates, str_values);

                    nnz_count++;
                    current_coordinates.clear();
                }
            }

            // Final check
            assert(nnz_count == nnz_header && "Matrix market header file doesn't match the number of nnz entries.");


            input_file.close();

        }
    }
}

#endif // MTXFILESIO_HPP