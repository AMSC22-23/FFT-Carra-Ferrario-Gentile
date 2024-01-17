#ifndef SPECTROGRAM_HPP
#define SPECTROGRAM_HPP

#include "ffft.hpp"

namespace spectrogram{
    
	template<class FFTStrategy>
    class SpectrogramGenerator
    {

        // Floating type retrieved by the one selected in the solver 
        using FloatingType = typename FFTStrategy::FloatTypeAlias;

        using CTensor_1D = fftcore::CTensorBase<FloatingType, 1>;
        using RTensor_1D = fftcore::CTensorBase<FloatingType, 1>;
        // Spectrogram output
        using RTensor_2D = fftcore::RTensorBase<FloatingType, 2>;

        public:
            SpectrogramGenerator(/* args */){};
            void spectrogram(CTensor_1D &signal, RTensor_2D &spectrogram,  int window_size);
            void spectrogram(RTensor_1D &signal, RTensor_2D &spectrogram,  int window_size);
        private:

    };

    /**
     * @brief 
     * 
     * @author Lorenzo Gentile
    */
	template<class FFTStrategy>
    void SpectrogramGenerator<FFTStrategy>::spectrogram(CTensor_1D &signal, RTensor_2D &spectrogram,  int window_size){

        FFTSolver<1, FloatingType> solver = std::make_unique<FFTStrategy>();
        
    }

    /**
     * @todo: I don't think this is needed
    */
    template<class FFTStrategy>
    void SpectrogramGenerator<FFTStrategy>::spectrogram(RTensor_1D &signal, RTensor_2D &spectrogram,  int window_size){
    }
}



#endif