#include "../src/realtime/Recorder.hpp"
#include "../src/core/Sequencer.hpp"
#include "../src/core/Identificator.hpp"
#include <vector>

using namespace zazamrealtime;
using namespace zazamcore;

int main(){

    while(true){
        Recorder<double> recorder;    
        std::vector<double> recording;
        recorder.record(recording);

        Sequencer sequencer("../local_dataset/hashes");
        Song sample, result;
        sequencer.sequence(recording, sample, false);

        Identificator identificator("../local_dataset/hashes");
        identificator.identify(sample.hash, result); 

        std::cout << "============================================================" << std::endl;
        std:cout << "Result: " << result.title << std::endl;
        std::cout << "============================================================" << std::endl;
    }
    return 0;
}