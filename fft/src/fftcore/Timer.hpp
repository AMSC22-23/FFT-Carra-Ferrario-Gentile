/**
 * @file Timer.hpp
 * @brief This file contains the Timer class, which is used to measure the
 *       execution time of different FFT implementations.
 * @details todo
 * @date 2023-12-04
**/

#ifndef TIMER_HPP
#define TIMER_HPP

#include <vector>
#include <chrono>
#include <iostream>



namespace fftcore{

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