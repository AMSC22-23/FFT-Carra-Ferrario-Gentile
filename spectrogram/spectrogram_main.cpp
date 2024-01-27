#include "ffft/spectrogram.hpp"

using namespace spectrogram;
using std::filesystem::path;

//settings
using FloatingType = double;
using Strategy = StockhamFFT<FloatingType>;
unsigned FRAME_LENGTH = 1024;
unsigned FRAME_STEP = 256;

path wav_directory = "../spectrogram/wav_samples";
path output_directory = "../spectrogram/output";

int main(int argc, char **argv){

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-w") {
            wav_directory = argv[++i];
        } else if (arg == "-o") {
            output_directory = argv[++i];
        } else if (arg == "--frame-length") {
            FRAME_LENGTH = std::stoi(argv[++i]);
        } else if (arg == "--frame-step") {
            FRAME_STEP = std::stoi(argv[++i]);
        } else {
            std::cout << "Usage: spectrogram_app.out [-w wav_directory] [-o output_directory] [--frame-length frame_length] [--frame-step frame_step]" << std::endl;
            return 1;
        }
    }

    //check if wav directory exists
    if(!std::filesystem::exists(wav_directory)){
        std::cout << "Wav directory at " << wav_directory << " does not exist." << std::endl;
        return 1;
    }

    // Check if output directory exists, if not create it
    if (!std::filesystem::exists(output_directory)) {
        std::cout << "Output directory does not exist, creating it..." << std::endl;
        std::filesystem::create_directory(output_directory);
    } else {
        // Clear output directory
        std::cout << "Output directory exists, clearing it..." << std::endl;
        std::filesystem::remove_all(output_directory);
        std::filesystem::create_directory(output_directory);
    }

    SpectrogramGenerator<Strategy> spectrogram_generator;

    // Now iterate over the directory contents
    for (const auto &entry : std::filesystem::directory_iterator(wav_directory)) {
        if (entry.path().extension() == ".wav") {
            spectrogram_generator.load_audio(entry.path(), FRAME_LENGTH, FRAME_STEP);
            spectrogram_generator.compute_spectrogram();
            path filename = output_directory / entry.path().filename().replace_extension(".txt");
            spectrogram_generator.get_last_spectrogram().write_to_file(filename);
        }
    }

    std::cout << "Done." << std::endl;

    return 0;
}