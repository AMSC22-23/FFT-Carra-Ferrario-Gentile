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
    Identificator identificator("../local_dataset/hashes");
    Sequencer sequencer("../local_dataset/hashes");
    Song sample,result;
    std::string sample_path = "../local_dataset/samples/redrum_raw.wav";


    sequencer.sequence(sample_path, sample, true);

    identificator.identify(sample.hash, result);

    std::cout << "============================================================" << std::endl;
    std:cout << "Result: " << result.title << std::endl;
    std::cout << "============================================================" << std::endl;
}