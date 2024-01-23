#ifndef SPECTROGRAM_GENERATOR_HPP
#define SPECTROGRAM_GENERATOR_HPP

#include <limits>
#include <type_traits>
#include "AudioFile.h"
#include "SpectrogramDataTypes.hpp"
#include "../fftcore/FFTSolver.hpp"
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

        //static assertion to check that FFTStrategy is 1D
        static_assert(std::is_base_of<FFT_1D<FloatingType>, FFTStrategy>::value, "To construct SpectrogramGenerator, FFTStrategy must be 1D");

        using Frame_t = Frame<FloatingType>;
        using Spectrogram_t = Spectrogram<FloatingType>;
        using CTensor_1D = fftcore::CTensorBase<1, FloatingType>;

        public:
            SpectrogramGenerator() : _fft_solver(std::make_unique<FFTStrategy>()), _is_loaded(false) {};

            /**
             * @brief Loads the signal from an existing tensor. The signal is divided into frames and stored inside the generator.
             * @param signal The signal to be transformed into a spectrogram.
             * @param frame_lenght The lenght of the frame window, in samples. It must be a power of 2.
             * @param frame_step The step between two consecutive frames, in samples. It default to frame_lenght.
             * @param pad_end If false, the last frame is discarded if it lies partially outside the signal. If true, the last frame is padded with zeros to match the frame lenght.
             */
            void load_audio(CTensor_1D &signal, TensorIdx frame_lenght, TensorIdx frame_step = 0, bool pad_end = false);

            /**
             * @brief Loads the signal from a .wav file. It relies on the AudioFile external header.
             * @param filename The path to the .wav file.
             * @param frame_lenght The lenght of the frame window, in samples. It must be a power of 2.
             * @param frame_step The step between two consecutive frames, in samples. It default to frame_lenght.
             * @param pad_end If false, the last frame is discarded if it lies partially outside the signal. If true, the last frame is padded with zeros to match the frame lenght.
             */
            void load_audio(std::filesystem::path filename, TensorIdx frame_lenght, TensorIdx frame_step = 0, bool pad_end = false);

            /**
             * @brief Computes the spectrogram from the loaded signal. All the spectrograms are stored in a vector inside the generator.
             * @param decibel_scale If true, the entries of the spectrogram are converted to decibels with the following transformation: x -> 10 * log10(x). The operation avoids numerical instability by adding a small epsilon to the argument of the logarithm. It defaults to false.
             * @param assume_real If true, the spectrogram is computed assuming that the signal is real, thus reducing the number of frequency bins by half. 
             * It defaults to true.
            */
            void compute_spectrogram(bool decibel_scale = true, bool assume_real = true);

            /**
             * @brief Returns a reference to all the spectrograms computed by the generator.
            */
            std::vector<Spectrogram_t>& get_spectrograms() { return _spectrograms; };

            /**
             * @brief Returns a reference to the last spectrogram computed by the generator.
            */
            Spectrogram_t& get_last_spectrogram() { return _spectrograms.back(); };
        private:

            //current parameters
            TensorIdx _signal_lenght, _frame_lenght;
            TensorIdx _num_frames;

            // FFT solver
            FFTSolver<1, FloatingType> _fft_solver;

            // This buffer will initially hold audio frames and then be transformed into intermediate FFT results
            std::vector<Frame_t> _transform_buffer;

            //This vector will hold the final spectrograms
            std::vector<Spectrogram_t> _spectrograms;

            bool _is_loaded;
    };

    template<class FFTStrategy>
    void SpectrogramGenerator<FFTStrategy>::load_audio(CTensor_1D &signal, TensorIdx frame_lenght, TensorIdx frame_step, bool pad_end){

        TensorIdx signal_lenght = signal.get_tensor().size();

        //if frame_step is not specified, it is set to frame_lenght
        if(frame_step == 0){
            frame_step = frame_lenght;
        }

        assert(signal_lenght >= frame_lenght && "Signal lenght must be greater than frame lenght");
        assert((frame_lenght & (frame_lenght - 1)) == 0 && "Frame lenght must be a power of 2");
        assert(frame_lenght >= frame_step && "Frame lenght must be greater than frame step");

        //compute the number of frames(excluding the last one if pad_end is false)
        TensorIdx num_frames = (signal_lenght - frame_lenght) / frame_step + 1;
        
        //clear the data vector and reserve space for the frames
        _transform_buffer.clear();
        _transform_buffer.reserve(num_frames);

        //load the frames into the data vector
        for(TensorIdx i = 0; i < num_frames; i++){
            _transform_buffer.push_back(Frame_t(signal, frame_lenght, frame_step, i));
        }

        //if pad_end is true, the last frame is loaded separately
        TensorIdx last_frame_start = num_frames * frame_step;
        if(pad_end && last_frame_start < signal_lenght){
            _transform_buffer.push_back(Frame_t(signal, frame_lenght, last_frame_start));
            num_frames++;
        }

        //store the current parameters
        _signal_lenght = signal_lenght;
        _frame_lenght = frame_lenght;
        _num_frames = num_frames;

        _is_loaded = true;
    }

    template<class FFTStrategy>
    void SpectrogramGenerator<FFTStrategy>::load_audio(std::filesystem::path filename, TensorIdx frame_lenght, TensorIdx frame_step, bool pad_end){

        using Complex = std::complex<FloatingType>;

        AudioFile<FloatingType> audio_file;
        bool success = audio_file.load(filename);
        assert(success && "Error loading audio file");

        //convert to mono if needed 
        if(audio_file.getNumChannels() == 2){
            for(auto &sample : audio_file.samples){
                sample[0] = (sample[0] + sample[1]) / 2;
            }
            audio_file.setNumChannels(1);
        }

        //convert to 1D tensor
        CTensor_1D signal(audio_file.getNumSamplesPerChannel());
        for(TensorIdx i = 0; i < signal.get_tensor().size(); i++){
            signal.get_tensor()(i) = Complex(audio_file.samples[0][i], 0);
        }

        load_audio(signal, frame_lenght, frame_step, pad_end);
    }

    template<class FFTStrategy>
    void SpectrogramGenerator<FFTStrategy>::compute_spectrogram(bool decibel_scale, bool assume_real){

        //check that a signal has been loaded
        assert(_is_loaded && "No signal loaded");

        //for slicing with Eigen
        using Index1D = std::array<TensorIdx, 1>;

        //compute the FFT of each frame
        for(auto &frame : _transform_buffer){
            _fft_solver.compute_fft(frame, FFT_FORWARD);
        }

        //compute the number of bins in the real and complex case
        unsigned int num_bins = assume_real ? _frame_lenght / 2 + 1 : _frame_lenght;

        //construct the spectrogram   
        Spectrogram_t spectrogram(num_bins, _num_frames);

        //copy the FFT results into the spectrogram and square the absolute value to get the power spectrum 
        Index1D offset = {0};
        Index1D extent = {num_bins};
        for(TensorIdx i = 0; i < _num_frames; i++){
            spectrogram.get_tensor().chip(i, 1) = _transform_buffer[i].get_tensor().slice(offset, extent).abs().square();
        }

        //convert to decibel scale if required
        if(decibel_scale){
            FloatingType epsilon = std::numeric_limits<FloatingType>::epsilon();
            FloatingType scale_factor = 10;
            auto power_to_decibel = [epsilon, scale_factor](FloatingType x){ return scale_factor * std::log10(x + epsilon); };
            spectrogram.get_tensor() = spectrogram.get_tensor().unaryExpr(power_to_decibel);
        }

        //spectrogram is moved into the vector
        _spectrograms.push_back(std::move(spectrogram));
    }
}

#endif //SPECTROGRAM_GENERATOR_HPP