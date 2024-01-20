#ifndef MTXFILESIOUTILS_HPP
#define MTXFILESIOUTILS_HPP

#include <iostream>
#include <vector>
#include <sstream>

/**
 * Utility methods for MtxFilesIOUtils
*/
namespace fftcore{
    namespace utils {

        std::vector<std::string> split_str_by_whitespace(const std::string& input) {
            std::istringstream iss(input);
            std::vector<std::string> result;

            // Use >> operator to extract words
            std::string word;
            while (iss >> word) {
                result.push_back(word);
            }

            return result;
        }

        /**
         * Load into params the properties of the data inside the mtx file.
         * @return false if the header is incorrect or incomplete, true in the other case.
        */
        bool load_mtx_tensor_file_settings(const string &header, bool &is_complex_data, bool is_symmetric){

            std::string help_header_template = "The header of the matrix file should match this template:\n%%MatrixMarket tensor coordinate <real/complex> <general/symmetric>\n";
            
            std::vector<std::string> h_sub_strings = split_str_by_whitespace(header);
            if(h_sub_strings.size() != 5){
                std::cerr << "Err: Matrix market tensor header is not well formatted." << std::endl;
                std::cerr << help_header_template;
                return false; 
            }
            if(h_sub_strings[0] != "%%MatrixMarket"){
                std::cerr << "Err: %%MatrixMarket label in header is missing." << std::endl;
                std::cerr << help_header_template;
                return false; 
            }
            if(h_sub_strings[1] != "tensor"){
                std::cerr << "Err: 'tensor' keyword is missing in the matrix market file header." << std::endl;
                std::cerr << help_header_template;
                return false; 
            }
            if(h_sub_strings[2] != "coordinate"){
                std::cerr << "Err: 'coordinate' keyword is missing in the matrix market file header." << std::endl;
                std::cerr << help_header_template;
                return false; 
            }
            
            if(h_sub_strings[3] == "real"){
                is_complex_data = false;
            }
            else if(h_sub_strings[3] == "complex"){
                is_complex_data = true;
            }else{
                std::cerr << "Err: 'real/complex' keyword is missing in the matrix market file header." << std::endl;
                std::cerr << help_header_template;
                return false;  
            }

            if(h_sub_strings[4] == "general"){
                is_symmetric = false;
            }
            else if(h_sub_strings[4] == "symmetric"){
                is_symmetric = true;
            }else{
                std::cerr << "Err: 'general/symmetric' keyword is missing in the matrix market file header." << std::endl;
                std::cerr << help_header_template;
                return false;  
            }

            return true;


        }

            template<typename DataType, int Rank>
            void _set_value(Eigen::Tensor<DataType, Rank>& tensor, std::vector<int> coordinates, std::vector<std::string> str_values){
                bool num_of_entries_check = str_values.size() == Rank + 1;
                assert(num_of_entries_check && "Error: real data is incomplete or badly formatted.");

                double current_value;
                current_value = stod(str_values.at(
                    Rank // Last index if number is real
                ));       

                // Setting the value into the tensor
                tensor(coordinates) = current_value;
            }

            template<int Rank>
            void tensor_set_value(Eigen::Tensor<std::complex<double>, Rank>& tensor, std::vector<int> coordinates, std::vector<std::string> str_values){
                bool num_of_entries_check = str_values.size() == Rank + 2;
                assert(num_of_entries_check && "Error: complex data is incomplete or badly formatted.");

                std::complex<double> current_value;
                current_value.real(stod(str_values.at(
                    Rank // Index of real part
                )));       
                current_value.imag(stod(str_values.at(
                    Rank+1 // Index of imaginary part
                )));

                // Setting the value into the tensor
                tensor(coordinates) = current_value;
            }
    }
}

#endif //MTXFILESIOUTILS_HPP