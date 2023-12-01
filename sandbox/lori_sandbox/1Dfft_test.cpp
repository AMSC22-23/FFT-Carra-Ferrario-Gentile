#include <iostream>
#include <complex>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <variant>

using namespace std::chrono;

enum TensorType
{
    EIGENVEC,
    STDVEC
};

template <typename DataType>
class FFTTest
{

    using ComplexStdVector = std::vector<std::complex<DataType>>;
    using ComplexEigenVector = Eigen::Matrix<std::complex<DataType>, Eigen::Dynamic, 1>;
    using Container = std::variant<ComplexStdVector, ComplexEigenVector>;

public:
    FFTTest(TensorType type) : type_(type)
    {
        // set random seed for reproducibility
        srand(0);
        ComplexEigenVector temp = ComplexEigenVector::Random(n_);

        switch(type){
            case TensorType::EIGENVEC :{
                input_ = temp;
                break;
            }
            case TensorType::STDVEC :{
                std::copy(temp.data(), temp.data() + temp.size(), std::back_inserter(std::get<ComplexStdVector>(input_)));
                break;
            }
            default: throw std::invalid_argument("Invalid tensor type.");
        }
    }

    void runTest(){
        if(debug){
            std::cout << "-- Input --" << std::endl;
            std::visit([this](auto &&arg)
                       {
                print_complex_vector(arg);
            },
                       input_);
        }

        high_resolution_clock::time_point start = high_resolution_clock::now();
        fft1d(input_);
        high_resolution_clock::time_point stop = high_resolution_clock::now();

        duration_ms_ = duration_cast<milliseconds>(stop - start);

        print_();

        if(debug){
            std::cout << "-- Output --" << std::endl;
            std::visit([this](auto &&arg)
                       {
                print_complex_vector(arg);
            },
                       input_);
        }
    }

    static void timeEigenFFT(){
        using ComplexEigenVector = Eigen::Matrix<std::complex<DataType>, Eigen::Dynamic, 1>;
        srand(0);
        ComplexEigenVector temp = ComplexEigenVector::Random(n_);

        Eigen::FFT<DataType> fft;
        high_resolution_clock::time_point start = high_resolution_clock::now();
        fft.fwd(temp, temp);
        high_resolution_clock::time_point stop = high_resolution_clock::now();

        milliseconds duration_ms = duration_cast<milliseconds>(stop - start);

        std::cout << "----- Eigen FFT Test -----" << std::endl;
        std::cout << "FFT length: " << n_ << std::endl;
        std::cout << "Underlying data type: " << typeid(DataType).name() << std::endl;
        std::cout << "Duration: " << duration_ms.count() << " milliseconds" << std::endl;
        std::cout << "---------------------" << std::endl << std::endl;
    }

private:
    // Function to perform the bit reversal of a given integer n
    unsigned int reverseBits(unsigned int n, int log2n)
    {
        unsigned int result = 0;
        for (int i = 0; i < log2n; i++)
        {
            if (n & (1 << i))
            {
                result |= 1 << (log2n - 1 - i);
            }
        }
        return result;
    }

    void fft1d(Container &data)
    {
        using Complex = std::complex<DataType>;
        std::visit([this](auto &&arg)
                   {
            int n = static_cast<int>(arg.size());
            int log2n = std::log2(n);

            if (n & (n - 1)) {
                throw std::invalid_argument("FFT length must be a power of 2.");
            }

            // Bit-reversal permutation
            for (unsigned int i = 0; i < n; ++i) {
                unsigned int rev = reverseBits(i, log2n);
                if (i < rev) {
                    std::swap(arg[i], arg[rev]);
                }
            }

            // Cooley-Tukey iterative FFT
            for (int s = 1; s <= log2n; ++s) {
                int m = 1 << s;         // 2 power s
                int m2 = m >> 1;        // m2 = m/2 -1
                Complex w(1, 0);
                Complex wm = exp(Complex(0, -2 * M_PI / m)); // w_m = e^(-2*pi/m)

                for (int j = 0; j < m2; ++j) {
                    for (int k = j; k < n; k += m) {
                        Complex t = w * arg[k + m2];
                        Complex u = arg[k];
                        arg[k] = u + t;
                        arg[k + m2] = u - t;
                    }
                    w *= wm;
                }
            } },
                   data);
    }

    void print_complex_vector(const ComplexStdVector &vec){
        for(auto entry : vec){
            std::cout << entry << std::endl;
        }
    }

    void print_complex_vector(const ComplexEigenVector &vec){
        for(auto entry : vec){
            std::cout << entry << std::endl;
        }
    }

    void print_()
    {
        std::cout << "----- FFT Test -----" << std::endl;
        std::cout << "FFT length: " << n_ << std::endl;
        std::cout << "Tensor type: " << (type_ == TensorType::EIGENVEC ? "Eigen Vector" : "Std Vector") << std::endl;
        std::cout << "Underlying data type: " << typeid(DataType).name() << std::endl;
        std::cout << "Duration: " << duration_ms_.count() << " milliseconds" << std::endl;
        std::cout << "---------------------" << std::endl << std::endl;
    }

    TensorType type_;
    Container input_;
    milliseconds duration_ms_;

    static constexpr bool debug = false;
    static constexpr int n_ = 1 << 22;
};

int main()
{

    FFTTest<float> test1(TensorType::EIGENVEC);
    FFTTest<float> test2(TensorType::STDVEC);
    FFTTest<double> test3(TensorType::EIGENVEC);
    FFTTest<double> test4(TensorType::STDVEC);

    test1.runTest();
    test2.runTest();
    test3.runTest();
    test4.runTest();

    FFTTest<float>::timeEigenFFT();
    FFTTest<double>::timeEigenFFT();

    return 0;
}