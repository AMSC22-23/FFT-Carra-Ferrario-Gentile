#include<unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

/**
 * IO utils for Eigen vectors, matrixes and tensors.
 * @TODO: Create a standard for saving tensors, for all ranks. (.mtx suits only vectors and matrixes.)
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
        std::cout << "Rows: " << spMat.rows() << std::endl;
        std::cout << "Cols: " << spMat.cols() << std::endl;

        target = Eigen::TensorMap<Eigen::Tensor<double, 2>>(denseMat.data(), denseMat.rows(), denseMat.cols());

    }


}