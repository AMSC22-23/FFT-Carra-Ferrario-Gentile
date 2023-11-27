#pragma once 

#include <iostream>
#include <vector>
#include <sstream>

/**
 * Utility methods for MtxFilesIOUtils
*/
namespace MtxFilesIOUtils{
    using namespace std;

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

        string help_header_template = "The header of the matrix file should match this template:\n%%MatrixMarket tensor coordinate <real/complex> <general/symmetric>\n";
        
        vector<string> h_sub_strings = split_str_by_whitespace(header);
        if(h_sub_strings.size() != 5){
            cerr << "Err: Matrix market tensor header is not well formatted." << endl;
            cerr << help_header_template;
            return false; 
        }
        if(h_sub_strings[0] != "%%MatrixMarket"){
            cerr << "Err: %%MatrixMarket label in header is missing." << endl;
            cerr << help_header_template;
            return false; 
        }
        if(h_sub_strings[1] != "tensor"){
            cerr << "Err: 'tensor' keyword is missing in the matrix market file header." << endl;
            cerr << help_header_template;
            return false; 
        }
        if(h_sub_strings[2] != "coordinate"){
            cerr << "Err: 'coordinate' keyword is missing in the matrix market file header." << endl;
            cerr << help_header_template;
            return false; 
        }
        
        if(h_sub_strings[3] == "real"){
            is_complex_data = false;
        }
        else if(h_sub_strings[3] == "real"){
            is_complex_data = false;
        }else{
            cerr << "Err: 'real/complex' keyword is missing in the matrix market file header." << endl;
            cerr << help_header_template;
            return false;  
        }

        if(h_sub_strings[4] == "general"){
            is_symmetric = false;
        }
        else if(h_sub_strings[4] == "symmetric"){
            is_symmetric = true;
        }else{
            cerr << "Err: 'general/symmetric' keyword is missing in the matrix market file header." << endl;
            cerr << help_header_template;
            return false;  
        }

        return true;


    }

}