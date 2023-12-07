#include <iostream>
#include <cmath>
#include <complex>
#include <array>
#include <chrono>
using namespace std;
using namespace std::chrono;

int main(){

    int max_s = 4, s2;
    int num_p = 8;
    int ka;
    for(int s=1; s<=max_s; s++){
        s2 = 1 << (s-1);
        for(int pi = 0; pi<num_p; pi++){
            ka = s2*(pi / s2) + pi;
            printf("s:%d s2:%d pi:%d  ka:%d\n", s, s2, pi, ka);
        }
        printf("\n\n");
    }

    
    std::complex<double> wm = exp(std::complex<double>(0, -2 * M_PI / 2)); // w_m = e^(2*pi/m)
    cout << wm;
    cout << wm*wm << endl;
    cout << std::pow(wm, 2) << endl;

    array<int,1024> src;
    array<int,1024> dst;
 
    auto start = high_resolution_clock::now();

    for(int i=0; i<1024; i++){
        dst[i] = src[i];
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);
 
    cout << "Time taken by function: "
         << duration.count() << " nanoseconds" << endl;
    return 0;
}