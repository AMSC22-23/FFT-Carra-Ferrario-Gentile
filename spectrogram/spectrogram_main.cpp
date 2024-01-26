#include "ffft/spectrogram.hpp"

using namespace spectrogram;

//modifiable parameters
using FloatingType = double;
using Strategy = StockhamFFT<FloatingType>;

//type aliases
using ComplexType = std::complex<FloatingType>;
using CTensor_1D = fftcore::CTensorBase<1, FloatingType>;

#define N_SIGNALS 3

using std::filesystem::path;
path wav_directory = "../wav_samples", output_directory = "output";

int main(void){

    std::vector<CTensor_1D> signals(N_SIGNALS);

    for(unsigned long i = 0; i < signals.size(); i++){
        signals[i] = CTensor_1D(1024);
        for(fftcore::TensorIdx j = 0; j < signals[i].get_tensor().size(); j++){
            signals[i].get_tensor()(j) = ComplexType(std::sin((i + 1) * (j + 1)), 0);
        }
    }

    SpectrogramGenerator<Strategy> spectrogram_generator;

    for(auto &signal : signals){
        spectrogram_generator.load_audio(signal, 128, 32);
        spectrogram_generator.compute_spectrogram();
    }

    auto spectrograms = spectrogram_generator.get_spectrograms();

    // Check if output directory exists, if not create it
    if (!std::filesystem::exists(output_directory)) {
        std::cout << "Output directory does not exist, creating it..." << std::endl;
        std::filesystem::create_directory(output_directory);
    }

    unsigned int file_index = 0;
    for(auto &spectrogram : spectrograms){
        path filename = output_directory / ("spectrogram_" + std::to_string(file_index) + ".txt");
        spectrogram.write_to_file(filename);
        file_index++;
    }

    /*------------------------------------------------------------------*/

    //check if wav directory exists
    assert(std::filesystem::exists(wav_directory) && "Wav directory does not exist");

    // Now iterate over the directory contents
    for (const auto &entry : std::filesystem::directory_iterator(wav_directory)) {
        if (entry.path().extension() == ".wav") {
            spectrogram_generator.load_audio(entry.path(), 1024, 256);
            spectrogram_generator.compute_spectrogram();
            path filename = output_directory / entry.path().filename().replace_extension(".txt");
            spectrogram_generator.get_last_spectrogram().write_to_file(filename);
        }
    }

    return 0;

}