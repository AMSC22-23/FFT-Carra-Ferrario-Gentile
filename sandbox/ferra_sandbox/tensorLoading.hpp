#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#define MATRIX_MARKET_TENSOR_TITLE "%%MatrixMarket tensor coordinate real general"
using namespace std;

std::vector<std::string> splitByWhitespace(const std::string& input) {
    std::istringstream iss(input);
    std::vector<std::string> result;

    // Use >> operator to extract words
    std::string word;
    while (iss >> word) {
        result.push_back(word);
    }

    return result;
}

template<typename DataType, int Rank>
void load_tensor_mtx(Eigen::Tensor<DataType, Rank> &tensor, std::string path){
    ifstream inputFile(path);

    if(!inputFile.is_open()){
        std::cerr << "Could not open the file at '" << path << "'." << endl;  
    }

    string line;
    getline(inputFile, line);
    if(line != MATRIX_MARKET_TENSOR_TITLE){
        cerr << "Matrix market header of the file is not complete." << endl;
        return;
    }

    int lineCount = 1;
    int nnz;
    vector<int> sizes;
    vector<string> strValues;
    
    while(getline(inputFile, line)){
        
        strValues = splitByWhitespace(line);

        // Reading tesor header
        if(lineCount == 1){
            if(strValues.size() != (Rank +1)){

                cerr << "Matrix market header file doesn't match the tensor rank.";
                return;
            }
            for(int i=0; i<strValues.size()-1; i++){
                sizes[i] = stoi(strValues[i]);
            }
            tensor = Eigen::Tensor<DataType, Rank>(sizes);
            nnz = stoi(strValues.back());   
        }

        // Reading coordinate
        

        lineCount++;
    }
    
    
    
}


