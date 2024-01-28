#ifndef SEQUENCER_HPP
#define SEQUENCER_HPP

#include "ffft/fftcore.hpp"
#include "ffft/spectrogram.hpp"
#include "utils/ZazamDataTypes.hpp"
#include "utils/MusicTensor.hpp"
#include "utils/Utils.hpp"
#include "HashGenerator.hpp"

#include <memory>
namespace zazamcore{

    /**
     * @brief Sequencer class takes care of sequencing audio data into hashes vector
     * through operations on its spectrogram.
     *
    */
    class Sequencer 
    {
        public:
            /**
             * @brief Contructor for Sequencer.
             * @param _output_path The path of the directory in which the sequenced songs 
             * hashes will be saved.
            */
            Sequencer(std::string _output_path):
                output_path(_output_path)
            {};
            /**
             * Constructor for Sequencer.
            */
            Sequencer(){};


            /**
             * @brief Sequence a song from a WAV or AIFF file.
             * @param path WAV or AIFF audio file path
             * @param result The resulting Song object
             * @param save_hash If true, save to the location specified in Sequencer.hpp
             * @returns A Song object with title and hash
            */
            void sequence(const std::string &path, Song &result, bool save_hash = true) const;

            /**
             * @brief Sequence a song from vector.
             * @param audio_vector WAV or AIFF audio file path
             * @param result The resulting Song object
             * @param save_hash If true, save to the location specified in Sequencer.hpp
             * @returns A Song object with title and hash
            */
            void sequence(const std::vector<double> &audio_vector, Song &result, bool save_hash = true) const;

        private:
            /**
             * @brief Path of the directory where the sequenced songs hashes will be saved. 
            */
            std::string output_path; 

            /**
             * @brief Sequence a song from a music tensor.
             * @param song_tensor 
             * @param result The resulting Song object
             * @param save_hash If true, save to the location specified in Sequencer.hpp
             * @returns A Song object with title and hash
            */
            void sequence(MusicTensor<double> &song_tensor, Song &result, std::string file_name, bool save_hash = true) const;
            

   };
}
#endif