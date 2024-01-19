#ifndef SPECTROGRAM_HPP
#define SPECTROGRAM_HPP

#include "../../include/ffft.hpp"
#include "SpectrogramDataTypes.hpp"

using namespace fftcore;

namespace spectrogram{
    
    /**
     * @brief SpectrogramGenerator transforms a 1D (complex or real) signal into a 2D spectrogram.
     * @tparam FFTStrategy The FFTStrategy to be used to compute the FFTs.
     * @author Lorenzo Gentile
    */
	template<class FFTStrategy>
    class SpectrogramGenerator
    {

        // Floating type retrieved by the one selected in the strategy
        using FloatingType = typename FFTStrategy::FloatTypeAlias;

        using Frame_t = Frame<FloatingType>;
        using Spectrogram_t = Spectrogram<FloatingType>;
        using CTensor_1D = fftcore::CTensorBase<1, FloatingType>;

        public:
            SpectrogramGenerator() : _fft_solver(std::make_unique<FFTStrategy>()){};

            void load_audio(CTensor_1D &signal, unsigned int frame_lenght, unsigned int frame_step = 0, bool pad_end = false);

            void compute_spectrogram(bool assume_real = true);

            /**
             * @brief Returns the spectrograms computed by the generator.
            */
            const std::vector<Spectrogram_t>& get_spectrograms() const { return _spectrograms; };
        private:

            //current parameters
            unsigned int _signal_lenght, _frame_lenght;
            unsigned int _num_frames;

            // FFT solver
            FFTSolver<1, FloatingType> _fft_solver;

            // This buffer will initially hold audio frames and then be transformed into intermediate FFT results
            std::vector<Frame_t> _data;

            //This vector will hold the final spectrograms
            std::vector<Spectrogram_t> _spectrograms;
    };

    /**
     * @brief Loads the signal into the generator and chops it into frames. For now it load the signal from an existing tensor.
     * @param signal The signal to be transformed into a spectrogram.
     * @param frame_lenght The lenght of the frame window, in samples. It must be a power of 2.
     * @param frame_step The step between two consecutive frames, in samples. It default to frame_lenght.
     * @param pad_end If false, the last frame is discarded if it lies partially outside the signal. If true, the last frame is padded with zeros to match the frame lenght.
     * @todo Add support for loading the signal from a file.
    */

    template<class FFTStrategy>
    void SpectrogramGenerator<FFTStrategy>::load_audio(CTensor_1D &signal, unsigned int frame_lenght, unsigned int frame_step, bool pad_end){

        unsigned int signal_lenght = signal.get_tensor().size();

        //if frame_step is not specified, it is set to frame_lenght
        if(frame_step == 0){
            frame_step = frame_lenght;
        }

        assert((signal_lenght & (signal_lenght - 1)) == 0 && "Signal lenght must be a power of 2");
        assert(signal_lenght >= frame_lenght && "Signal lenght must be greater than frame lenght");
        assert((frame_lenght & (frame_lenght - 1)) == 0 && "Frame lenght must be a power of 2");
        assert(frame_lenght >= frame_step && "Frame lenght must be greater than frame step");

        //compute the number of frames(excluding the last one if pad_end is false)
        unsigned int num_frames = (signal_lenght - frame_lenght) / frame_step + 1;
        
        _data.clear();
        _data.reserve(num_frames);

        for(unsigned int i = 0; i < num_frames; i++){
            _data.push_back(Frame_t(signal, frame_lenght, frame_step, i));
        }

        unsigned int last_frame_start = num_frames * frame_step;

        if(pad_end && last_frame_start < signal_lenght){
            _data.push_back(Frame_t(signal, frame_lenght, last_frame_start));
            num_frames++;
        }

        _signal_lenght = signal_lenght;
        _frame_lenght = frame_lenght;
        _num_frames = num_frames;
    }

    /**
     * @brief Computes the spectrogram from the loaded signal. All the spectrograms are stored in a vector inside the generator.
     * @param assume_real If true, the spectrogram is computed assuming that the signal is real, thus reducing the number of frequency bins by half. 
     * It defaults to true.
    */
    template<class FFTStrategy>
    void SpectrogramGenerator<FFTStrategy>::compute_spectrogram(bool assume_real){

        //for slicing with Eigen
        using Index1D = std::array<Eigen::Index, 1>;

        //compute the FFT of each frame
        for(auto &frame : _data){
            _fft_solver.compute_fft(frame, FFT_FORWARD);
        }

        //compute the number of bins in the real and complex case
        unsigned int num_bins = assume_real ? _frame_lenght / 2 + 1 : _frame_lenght;

        //construct the spectrogram   
        Spectrogram_t spectrogram(num_bins, _num_frames);

        //copy the FFT results into the spectrogram
        Index1D offset = {0};
        Index1D extent = {num_bins};
        for(unsigned int i = 0; i < _num_frames; i++){
            //copy the first half of each fft result and take the absolute value
            spectrogram.get_tensor().chip(i, 1) = _data[i].get_tensor().slice(offset, extent).abs();
        }

        //spectrogram is moved into the vector
        _spectrograms.push_back(std::move(spectrogram));
    }
}

#endif //SPECTROGRAM_HPP