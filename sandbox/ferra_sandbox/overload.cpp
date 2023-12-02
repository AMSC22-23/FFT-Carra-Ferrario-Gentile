#include <complex>
#include <iostream>

using namespace std;

template<typename DataType>
void print(DataType value){
    cout << value;
}

void print(std::complex<double> value){
    cout<< value.real() << ", " << value.imag();
}

int main(){

    std::complex<double> value = {1.0, 2.0};
    print(value);
    return 0;
}