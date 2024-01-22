#include "../src/core/HashGenerator.hpp"
#include "../src/core/Identificator.hpp"
#include "../src/core/Sequencer.hpp"
#include <iostream>
#include <vector>
#include <filesystem>

using namespace std;
using namespace zazamcore;
int main(){

    Sequencer sequencer("../local_dataset/hashes");
    Song song ;

    std::string wav_path = "../local_dataset/wav";
    for (const auto & entry : std::filesystem::directory_iterator(wav_path)){
        std::cout << entry.path() << std::endl;
        std::string path = entry.path().string();
        sequencer.sequence(path, song);
    }
}