#ifndef SPECTROGRAM_DATA_TYPES_HPP
#define SPECTROGRAM_DATA_TYPES_HPP

#include <fstream>
#include <iomanip>
#include "../../include/ffft.hpp"

namespace spectrogram{

    /**
     * @brief This class represents a single frame of a signal. It extends TensorFFTBase.
     * It provides two constructors: one for frames that lie completely inside the signal and one for frames that lie partially outside the signal.
    */
    template<typename FloatingType>
    class Frame : public CTensorBase<1, FloatingType>
    {
        //for slicing with Eigen
        using Index1D = std::array<Eigen::Index, 1>;

        public:

            /**
             * @brief Constructor for frames that lie completely inside the signal. If pad_end is false, only this constructor should be called.
            */
            Frame(CTensorBase<1, FloatingType> &signal, unsigned int frame_lenght, unsigned int frame_step, unsigned int frame_index) : CTensorBase<1, FloatingType>(frame_lenght){

                Index1D offset = {frame_index * frame_step};
                Index1D extent = {frame_lenght};

                this->get_tensor() = signal.get_tensor().slice(offset, extent);
            }; 

            /**
             * @brief Constructor for frames that lie partially outside the signal. If pad_end is true, this constructor is called for the last frame. 
             * @note The last frame is padded with zeros to the right to match the frame lenght (the base constructor initializes the tensor with zeros)
            */
            Frame(CTensorBase<1, FloatingType> &signal, unsigned int frame_lenght, unsigned int last_frame_start) : CTensorBase<1, FloatingType>(frame_lenght){

                unsigned int last_frame_lenght = signal.get_tensor().size() - last_frame_start;

                memcpy(this->get_tensor().data(), signal.get_tensor().data() + last_frame_start, last_frame_lenght * sizeof(FloatingType));
            
            }
    };


    /**
     * @brief This class represents a spectrogram. It wraps a 2D real Eigen tensor.
     * @todo Add utils method (plot, etc...)
    */
    template<typename FloatingType>
    class Spectrogram{

        public:
            Spectrogram(unsigned int num_frames, unsigned int frame_lenght) : _tensor(Eigen::Tensor<FloatingType, 2>(num_frames, frame_lenght)){};

            void write_to_file(std::string filename);

            //void plot(); TODO?

            Eigen::Tensor<FloatingType, 2>& get_tensor() { return _tensor; };
        private:
            Eigen::Tensor<FloatingType, 2> _tensor;
    };

    /**
     * @brief Writes the spectrogram to a file. The file format is the following:
     * - The first line contains the number of bins and the number of frames, separated by a space.
     * - The following lines contain the spectrogram data, one row per line. The values are written in scientific notation with 2 significant digits.
     * @param filename The name of the file to write to.
    */
    template<typename FloatingType>
    void Spectrogram<FloatingType>::write_to_file(std::string filename){

        std::ofstream file(filename);
        file << std::scientific << std::setprecision(2);

        if(file.is_open()){

            file << _tensor.dimension(0) << " " << _tensor.dimension(1) << std::endl;

            for(int i = 0; i < _tensor.dimension(0); i++){
                for(int j = 0; j < _tensor.dimension(1); j++){
                    file << _tensor(i, j) << " ";
                }
                file << std::endl;
            }
            file.close();
            std::cout << "Spectrogram written to file " << filename << std::endl;
        }
        else{
            std::cerr << "Unable to open file " << filename << std::endl;
        }
    }
}

#endif // SPECTROGRAM_DATA_TYPES_HPP