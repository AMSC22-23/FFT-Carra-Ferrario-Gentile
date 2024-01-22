#ifndef UTILS_HPP
#define UTILS_HPP

#include "ZazamDataTypes.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>

namespace zazamcore{
    namespace utils {

        template<typename Scalar> 
        int find_max_element_index(const Vector<Scalar> &input, Scalar &max_element){
            auto max = std::max_element(input.data(), input.data()+input.size());
            return std::distance(input.data(), max);
        }
        
        template<typename Scalar> 
        int find_max_element_index(const Vector<Scalar> &input){
            auto max = std::max_element(input.data(), input.data()+input.size());
            return std::distance(input.data(), max);
        }
        
        template<typename Scalar> 
        int find_max_element_index(const std::vector<Scalar> &input, Scalar &max_element){
            auto max = std::max_element(input.begin(), input.end()); 
            return std::distance(input.begin(), max);
        }
        
        template<typename Scalar> 
        int find_max_element_index(const std::vector<Scalar> &input){
            auto max = std::max_element(input.begin(), input.end());
            return std::distance(input.begin(), max);
        }

        template<typename T>
        void print_std_vector(const std::vector<T> &input){
            std::cout << "[";
            for(int i=0; i<input.size(); i++){
                std::cout << " " << input[i] << ",";
            }
            std::cout << "]" <<std::endl;
        }
        template<typename T>
        void std_to_complex_eigen_vector(const std::vector<T> &vector, Vector<std::complex<T>> &eigen_vector, size_t begin, e_index dimension){

            std::complex<T> val(0,0);
            for(e_index i=0; i<dimension; i++){
                val.real(vector[begin+i]);
                eigen_vector(i) = val; 
            } 
        } 
        
        /**
         * @brief Save a 1-ranked eigen vector to file according to the this app standard notations. 
         * @see MtxFilesIO in ffft library
         * @param vector The vector to save.
         * @param path The location where to save the file.
        */
        template<typename T>
        void save_real_vector(Vector<T> &vector, std::string path){
            // Save the tensor to Matrix Market format
            std::ofstream file(path);
            if (file.is_open()) {
                file << "%%MatrixMarket tensor coordinate real general" << std::endl;
                file << vector.size() << " " << vector.size() << std::endl;

                // Iterate over the tensor and write each element to the file
                for (int i = 0; i < vector.size(); ++i) {
                    file << i+1 << " ";
                    file << vector(i) << std::endl;
                }

                file.close();
                std::cout << "Tensor saved to " << path << std::endl;
            } else {
                std::cerr << "Unable to open file: " << path << std::endl;
            }
        }
        
        /**
         * @brief Find the mode of the vector, i.e. the most repeated element.
         * @param vector
         * @param mode The most repeated element 
         * @param max_occurrences The most repeated element number of occurrences 
        */
        template<typename T>
        void mode_of_vector(const std::vector<T> &vector, T &mode, T &max_occurrences){
        
            assert(vector.size() > 0);

            std::unordered_map<T, int> frequencyMap;

            // Count the frequency of each number in the vector
            for (auto &num : vector) {
                frequencyMap[num]++;
            }

            // Find the mode (number with the highest frequency)
            mode = vector[0]; // Initialize mode with the first element
            max_occurrences = frequencyMap[mode];

            for (const auto& entry : frequencyMap) {
                if (entry.second > max_occurrences) {
                    mode = entry.first;
                    max_occurrences = entry.second;
                }
            }
        
        }

    }
} 

#endif