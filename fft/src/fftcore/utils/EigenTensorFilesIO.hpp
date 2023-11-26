#include<unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

/**
 * IO utils for Eigen vectors, matrixes and tensors.
 * @TODO: Create a standard for saving tensors, for all ranks. 
 * 
 * @author Daniele Ferrario
*/
namespace EigenTensorFilesIO{

    /*
    template<typename DataType, int Dim>
    void save_vector_mtx(std::string path, const Eigen::Matrix<DataType, Dim, 1> vec){
        // Eigen::saveMarketVector(b, "./rhs.mtx");
        FILE* out = fopen(path.c_str(),"w");
        fprintf(out,"%%%%MatrixMarket vector coordinate real general\n");
        fprintf(out,"%d\n", vec.rows());
        for (int i=0; i<vec.rows(); i++) {
            fprintf(out,"%d %f\n", i , vec(i));
        }
        fclose(out);
    }

    template<typename DataType, int Rows, int Columns>
    void save_matrix_mtx(std::string path, const Eigen::Matrix<DataType, Rows, Columns> mat){
        std::string matrixFileOut(path);
        Eigen::saveMarket(mat, matrixFileOut);
    }
    */

    /**
     * Loads a 2d .mtx file into a two-ranked Eigen::Tensor object.
     * 
     * @TODO: Unfortunally, to load a Dense matrix (and then convert it to a Tensor), 
     * we must pass it through a Sparse Matrix first since loadMarketDense is not present
     * in the current version of Eigen.
     * 
     * @param target The Eigen::Tensor object
     * @param path the file path
    */
    template<typename DataType>
    void load_2d_mtx(Eigen::Tensor<DataType, 2> &target, const std::string &path){


        

        Eigen::SparseMatrix<double> spMat;
        Eigen::MatrixXd denseMat;
        Eigen::loadMarket(spMat, path);
        denseMat = Eigen::MatrixXd(spMat);

        target = Eigen::TensorMap<Eigen::Tensor<double, 2>>(denseMat.data(), denseMat.rows(), denseMat.cols());

    }

    /**
     * Loads a 1d .mtx file into a one-ranked Eigen::Tensor object.
     * 
     * @TODO: Unfortunally, to load a Dense matrix (and then convert it to a Tensor), 
     * we must pass it through a Sparse Matrix first since loadMarketDense is not present
     * in the current version of Eigen.
     * 
     * @param target The Eigen::Tensor object
     * @param path the file path
    */
    template<typename DataType>
    void load_1d_mtx(Eigen::Tensor<DataType, 1> &target, const std::string &path){

        Eigen::SparseMatrix<double> spMat;
        Eigen::MatrixXd denseMat;
        Eigen::loadMarket(spMat, path);
        denseMat = Eigen::MatrixXd(spMat);

        target = Eigen::TensorMap<Eigen::Tensor<double, 1>>(denseMat.data(), denseMat.rows(), denseMat.cols());

    }

    /**
     * Load a x-dimensional .mtx file into a x-ranked Eigen::Tensor object.
     * @TODO: to implement
    */
    template<typename DataType, int Rank>
    void load_tensor_mtx(Eigen::Tensor<DataType, Rank> &target, const std::string &path){

    }

}