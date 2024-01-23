# Zazam

Zazam is a simple music detection library built in C++, based on FFFT library.

## Required software 
This library depends on two libraries: `ffft` and `PortAudio`.

`ffft` library takes care of Fourier transforms, and in particular the spectrograms of songs.

`PortAudio` is involved in real-time recognition by handling the device audio input signal.

## Usage - Abstract

### Populating the dataset 
The songs are mapped and stored as vectors of hashes, which are calculated based on the most prominient frequencies at each time step. To generate them, and so to populate the dataset, we use the class `Sequencer`, which takes care of generating the vector of hashes from the spectrogram of the song.
### Identifying samples
Once that our dataset is ready, a sample of a song can be identified through the `Identifier` class. The main method will transform the sample into its corresponding vector of hashes. 
Then, the method will compare it to the ones contained in the dataset, calculating a matching score and eventually picking the song with the highest one.
## Usage - Code 
### Populating the dataset
The file `examples/sequence.cpp` provides the complete implementation of the populating phase, which is essentially based on performing the sequencing operation on each of the WAV of AIFF files provided:
```c++
// Initialize sequencer, indicating the hashes output path.
Sequencer sequencer("../local_dataset/hashes");
// Provide the path of the WAV or AIFF audio file.
std::string wav_path = "../local_dataset/wav/pink_floyd.wav";
// Song struct which will contain the title and the hash.
Song song;
// Sequence, save to file and load results in song object.
sequencer.sequence(wav_path, song);
```
### Identifying samples
The file `examples/identify.cpp` provides the implementation of sample recognition from a file, while `examples/record.cpp` involves real-time recognition using the default audio input source of the device. Essentially, the sample audio vector is mapped to its hash vector and then it's handled by the identify method:
```c++
std::vector<double> recording;

// Record for X seconds from microphone.
Recorder<double> recorder;    
recorder.record(recording);

Song sample, result;

// No need to indicate the output path because
// we do not want to the save the sample hash to file.
Sequencer sequencer;
sequencer.sequence(recording, sample, false);

// Provide the dataset path.
Identificator identificator("../local_dataset/hashes");
// Identify the sample.
identificator.identify(sample.hash, result); 
```
## Notes on performance
The library works very well overall. As expected, peak performances are reached with a good microphone and so a good sample quality. Moreover, the algorithm works the best with samples which are uniquely associatable to a certain song, rather then simple patterns which do not contain sensible variations in frequency domain.
## References 
The algorithm has been adapted from Joseph DeChicchis's paper 'A Music Identification Algorithm which Utilizes the Fourier
Analysis of Audio Signals'.
