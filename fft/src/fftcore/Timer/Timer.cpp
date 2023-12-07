#include "Timer.hpp"

namespace fftcore{

    void Timer::start(){
        _current_start = std::chrono::high_resolution_clock::now();
    }

    void Timer::stop(){
        auto current_stop = std::chrono::high_resolution_clock::now();
        _durations.push_back(current_stop - _current_start);
    }

    void Timer::reset(){
        _durations.clear();
    }

    void Timer::print() const{
        
        std::cout << "\n---- Timings ----" << std::endl;
        for(size_t i = 0; i < _durations.size(); ++i){
            
            double time = _durations[i].count();
            if(time < 1e-6){
                std::cout << i+1 << ": " << time*1e9 << " ns" << std::endl;
            }else if(time < 1e-3){
                std::cout << i+1 << ": " << time*1e6 << " us" << std::endl;
            }else if(time < 1){
                std::cout << i+1 << ": " << time*1e3 << " ms" << std::endl;
            }else{
                std::cout << i+1 << ": " << time << " s" << std::endl;
            }
        }
        std::cout << "-----------------\n" << std::endl;
    }

}