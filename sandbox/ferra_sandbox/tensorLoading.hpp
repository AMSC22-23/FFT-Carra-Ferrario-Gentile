#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#define MATRIX_MARKET_TENSOR_TITLE "%%MatrixMarket tensor coordinate real general"
using namespace std;

std::vector<std::string> splitString(const std::string& input, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream tokenStream(input);
    std::string token;

    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
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
    int[Rank] sizes;
    vector<string> strValues;
    while(getline(inputFile, line)){
        if(lineCount == 1){
            strValues = splitString(line, " ");
            if(strValues.size() != (Rank +1)){
                cerr << "Matrix market header file doesn't match the tensor rank.";
                return;
            }
            for(int i=0; i<strValues.size()-1; i++){
                sizes[i] = strValues[i];
            }
            nnz = strValues.back();

        }
    }
    
    
}


