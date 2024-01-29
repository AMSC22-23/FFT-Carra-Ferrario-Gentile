#include "../src/core/HashGenerator.hpp"
#include "../src/core/Identificator.hpp"
#include "../src/core/Sequencer.hpp"
#include "../src/realtime/Recorder.hpp"

#include <iostream>
#include <vector>
#include <filesystem>

using namespace std;
using namespace zazamcore;

int main(){
    Identificator identificator("../zazam/examples/dataset/hashes");
    Sequencer sequencer("../zazam/examples/dataset/hashes");
    Song sample,result;
    std::string sample_path = "../zazam/examples/dataset/samples/Paranoid_5s_noise.wav";


    sequencer.sequence(sample_path, sample, false);

    identificator.identify(sample.hash, result);

    std::cout << "============================================================" << std::endl;
    std:cout << "Result: " << result.title << std::endl;
    std::cout << "============================================================" << std::endl;
}