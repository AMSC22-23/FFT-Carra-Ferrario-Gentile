#include "../../ffft/src/spectrogram/Spectrogram.hpp"

#include <cmath>

using namespace spectrogram;
using CTensor_1D = fftcore::CTensorBase<1, double>;

int main(void){

    std::vector<CTensor_1D> signals(10);

    for(unsigned int i = 0; i < signals.size(); i++){
        signals[i] = CTensor_1D(1024);
        for(unsigned int j = 0; j < signals[i].get_tensor().size(); j++){
            signals[i].get_tensor()(j) = std::complex<double>(std::sin(i * j), 0);
        }
    }

    SpectrogramGenerator<SequentialFFT<>> spectrogram_generator;

    for(auto &signal : signals){
        spectrogram_generator.load_audio(signal, 128, 32);
        spectrogram_generator.compute_spectrogram();
    }

    auto spectrograms = spectrogram_generator.get_spectrograms();

    unsigned int file_index = 0;
    for(auto &spectrogram : spectrograms){
        spectrogram.write_to_file("output/" + std::to_string(file_index) + ".txt");

        file_index++;
    }

    return 0;

}