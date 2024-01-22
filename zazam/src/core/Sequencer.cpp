#include "Sequencer.hpp"

using namespace zazamcore;

void Sequencer::sequence(const std::string &path, Song &result, bool save_hash) const{

    if(save_hash){
        assert(!output_path.empty() && "The hashes saving directory has not been declared.");
    }


    // Retrieve filename from path string
    std::string base_filename = path.substr(path.find_last_of("/\\") + 1);
    size_t pos = base_filename.find('.');
    if (pos != std::string::npos)
        base_filename = base_filename.substr(0, pos);
    // ----------------------------------

    // Initialize the tensor wrapper which will load the file
    MusicTensor<double> song_tensor(path);
    sequence(song_tensor, result, base_filename, save_hash);
 
}

void Sequencer::sequence(const std::vector<double> &audio_vector, Song &result, bool save_hash) const{

    if(save_hash){
        assert(!output_path.empty() && "The hashes saving directory has not been declared.");
    }

    /** @todo: can be improved */
    std::string file_name = "vector";

    MusicTensor<double> song_tensor(audio_vector);
    sequence(song_tensor, result, file_name, save_hash);
    
}

void Sequencer::sequence(MusicTensor<double> &song_tensor, Song &result, std::string file_name, bool save_hash) const{

    // This matrix will contain the spectrogram
    Matrix_d song_matrix;

    // Initialize the FFFT solver
    FFTSolver<1, double> solver(std::make_unique<OmpFFT<double>>());
    int sequence_window = 1 << 11;
    
    // Calculate spectogram
    spectrogram::SpectrogramGenerator<fftwFFT<>> spectrogram_generator;
    spectrogram_generator.load_audio(song_tensor, sequence_window, sequence_window);
    spectrogram_generator.compute_spectrogram();
    auto spectrograms = spectrogram_generator.get_spectrograms(); 
    song_matrix = spectrograms.at(0).get_tensor(); // We calculate only one spectrogram

    // Create hash from spectrogram
    HashGenerator hash_generator;
    hash_generator.generate(song_matrix, result.hash, true);

    // --- Output saving ---


    if(save_hash)
        zazamcore::utils::save_real_vector(result.hash, output_path+"/"+file_name+".mtx");


    result.title = output_path;
}
