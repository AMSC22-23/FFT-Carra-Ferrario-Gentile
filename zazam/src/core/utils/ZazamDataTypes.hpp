#ifndef ZAZAM_DATATYPES_HPP
#define ZAZAM_DATATYPES_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>

namespace zazamcore{

    /**
     *  In this application, vectors are one-ranked Eigen tensors to mantain coherence 
     *  with FFFT data types. 
    */
    template<typename ScalarType>    
    using Vector = Eigen::Tensor<ScalarType, 1>;

    /**
     *  In this application, Matrices are two-ranked Eigen tensors to mantain coherence 
     *  with FFFT data types. 
    */
    template<typename ScalarType>    
    using Matrix = Eigen::Tensor<ScalarType, 2>;


    using Matrix_d = Matrix<double>;
    using Vector_d = Vector<double>;

    
    using Vector_ui = Vector<unsigned int>;

    using e_index = Eigen::Index; 

    struct Song{
        std::string title;
        Vector_ui hash;
    };

    // The range of frequencies to subdivide 
    static const std::pair<int, int> KeyPointsRange = {40,200};

    // The number of key points to use
    static const int KeyPointsNumber = 4;
}

#endif