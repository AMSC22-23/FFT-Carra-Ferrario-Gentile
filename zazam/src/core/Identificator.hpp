#ifndef IDENTIFICATOR_HPP
#define IDENTIFICATOR_HPP

#include "ffft/fftcore.hpp"
#include "utils/ZazamDataTypes.hpp"
#include "utils/Utils.hpp"
#include <vector>


namespace zazamcore{

    /**
     * @brief Identificator class takes care of the actual identification on the songs
     * by applying the matching algorithms on the dataset.
     * 
    */
    class Identificator{

        public:
            /**
             * Constructor for Identificator.
             * @param _hashes_dataset_path The dataset hashes directory path 
            */
            Identificator(const std::string _hashes_dataset_path) :
                hashes_dataset_path(_hashes_dataset_path)
            {};

            /**
             * @brief Identify the the song by a sample.
             * @param sample_hash The hashed sample vector 
             * @param result The song object with the highest matching score
            */
            void identify(const Vector_ui &sample_hash, Song &result) const;

        private:
            /**
             * @brief Normalize an hash vector according to the algorithm specifications.
             * @param hash The hash vector to normalize
            */
            void normalize_and_round(Vector_ui &hash) const;
            /**
             * @brief Calculate the scores of the matches between the sample and a song. 
             * @param song_hash
             * @param sample_hash
             * @param matches_vector The matches vector to fill with the scores. 
            */
            void calculate_matches_scores(const Vector_ui &song_hash, const Vector_ui &sample_hash, std::vector<int> &matches_vector) const;
            /**
             *  The dataset hashes directory path 
            */
            std::string hashes_dataset_path;
    };
}
#endif