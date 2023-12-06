#ifndef TIMER_HPP
#define TIMER_HPP

#include <vector>
#include <chrono>
#include <iostream>



namespace fftcore{

    /**
     * @brief This class is used to measure the execution time of different FFT implementations.
     * @date 2023-12-06
     * @author Lorenzo Gentiile
     **/
    class Timer{

        public:

        using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
        using duration = std::chrono::duration<double>;

        Timer() = default;
        ~Timer() = default;

        void start();
        void stop();
        void reset();

        void print() const;

        private: 
            time_point _current_start;
            std::vector<duration> _durations;
    };

}

#endif // TIMER_HPP