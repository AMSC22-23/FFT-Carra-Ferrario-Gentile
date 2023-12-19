#ifndef FFTWFFT_HPP
#define FFTWFFT_HPP

#include "../../FFTSolver.hpp"
#include "../../utils/FFTUtils.hpp"
#include <type_traits>
#include <fftw3.h>

namespace fftcore
{
    /**
     * @brief This class provides an FFT implementation based on the FFTW library.
     * It is used for benchmarking purposes. It doesn't support R2C in dimensions higher than 1.
     * @todo For now R2C has only forward mode. Consider if and how to implement inverse (C2R ?)
     * @author Lorenzo Gentile
     * @date 2023-12-06
    */
    template <typename FloatingType = double>
    class fftwFFT : public FFT_1D<FloatingType>,
                    public FFT_2D<FloatingType>,
                    public FFT_3D<FloatingType>
    {
    public:
        using typename FFT_1D<FloatingType>::RTensor_1D;
        using typename FFT_1D<FloatingType>::CTensor_1D;

        using typename FFT_2D<FloatingType>::RTensor_2D;
        using typename FFT_2D<FloatingType>::CTensor_2D;

        using typename FFT_3D<FloatingType>::RTensor_3D;
        using typename FFT_3D<FloatingType>::CTensor_3D;

        using Complex = std::complex<FloatingType>;

        fftwFFT()
        {
            static_assert(std::is_same<FloatingType, double>::value, "Only double precision is supported");
        };

        void fft(const CTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(const RTensor_1D &, CTensor_1D &, FFTDirection) const;

        void fft(CTensor_1D &, FFTDirection) const;

        void fft(const CTensor_2D &, CTensor_2D &, FFTDirection) const;

        void fft(const RTensor_2D &, CTensor_2D &, FFTDirection) const;

        void fft(CTensor_2D &, FFTDirection) const;

        void fft(const CTensor_3D &, CTensor_3D &, FFTDirection) const;

        void fft(const RTensor_3D &, CTensor_3D &, FFTDirection) const;

        void fft(CTensor_3D &, FFTDirection) const;
        ~fftwFFT() = default;
    };

    template <typename FloatingType>
    void fftwFFT<FloatingType>::fft(const CTensor_1D &input, CTensor_1D &output, FFTDirection fftDirection) const
    {
        memcpy(output.data(), input.data(), input.size() * sizeof(Complex));
        fft(output, fftDirection);
    };

    template <typename FloatingType>
    void fftwFFT<FloatingType>::fft(const RTensor_1D &input, CTensor_1D &output, FFTDirection) const
    {

        int n = input.size();

        FloatingType *in = nullptr;
        fftw_complex *out = nullptr;

        in = const_cast<FloatingType *>(input.data());
        out = reinterpret_cast<fftw_complex *>(output.data());

        fftw_plan plan = fftw_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);

        fftw_execute(plan);

        // Adding conjugate symmetric components (fftw does not do it by default)
        for (int i = n / 2 + 1; i < n; ++i)
        {
            output(i) = std::conj(output(n - i));
        }

        fftw_destroy_plan(plan);
    };

    template <typename FloatingType>
    void fftwFFT<FloatingType>::fft(CTensor_1D &input_output, fftcore::FFTDirection fftDirection) const
    {
        //fft(const_cast<const CTensor_1D &>(input_output), input_output, fftDirection);

        int n = input_output.size();

        fftw_complex *in = nullptr;
        int direction = fftDirection == FFT_FORWARD ? FFTW_FORWARD : FFTW_BACKWARD;

        in = reinterpret_cast<fftw_complex *>(input_output.data());

        fftw_plan plan = fftw_plan_dft_1d(n, in, in, direction, FFTW_ESTIMATE);

        fftw_execute(plan);

        // Scaling (fftw does not scale by default)
        if (direction == FFTW_BACKWARD)
        {
            input_output = input_output * Complex(1.0 / n, 0);
        }

        fftw_destroy_plan(plan);

    };

    template <typename FloatingType>
    void fftwFFT<FloatingType>::fft(const CTensor_2D &input, CTensor_2D &output, FFTDirection fftDirection) const
    {

        memcpy(output.data(), input.data(), input.size() * sizeof(Complex));
        fft(output, fftDirection);
    };

    template <typename FloatingType>
    void fftwFFT<FloatingType>::fft(const RTensor_2D &, CTensor_2D &, FFTDirection) const
    {
        throw NotSupportedException("Operation is not supported");
    };

    template <typename FloatingType>
    void fftwFFT<FloatingType>::fft(CTensor_2D &input_output, FFTDirection fftDirection) const
    {
        fft(const_cast<const CTensor_2D &>(input_output), input_output, fftDirection);

        int n = input_output.dimension(0);
        int m = input_output.dimension(1);

        fftw_complex *in = nullptr;
        int direction = fftDirection == FFT_FORWARD ? FFTW_FORWARD : FFTW_BACKWARD;

        in = reinterpret_cast<fftw_complex *>(input_output.data());

        fftw_plan plan = fftw_plan_dft_2d(m, n, in, in, direction, FFTW_ESTIMATE);

        fftw_execute(plan);

        // Scaling (fftw does not scale by default)
        if (direction == FFTW_BACKWARD)
        {
            input_output = input_output * Complex(1.0 / (n * m), 0);
        }

        fftw_destroy_plan(plan);
    };

    template <typename FloatingType>
    void fftwFFT<FloatingType>::fft(const CTensor_3D& input, CTensor_3D& output, FFTDirection fftDirection) const
    {
        memcpy(output.data(), input.data(), input.size() * sizeof(Complex));
        fft(output, fftDirection);
    };

    template <typename FloatingType>
    void fftwFFT<FloatingType>::fft(const RTensor_3D &, CTensor_3D &, FFTDirection) const
    {
        throw NotSupportedException("Operation is not supported");
    };

    template <typename FloatingType>
    void fftwFFT<FloatingType>::fft(CTensor_3D& input_output, FFTDirection fftDirection) const
    {
        fft(const_cast<const CTensor_3D &>(input_output), input_output, fftDirection);

        int n = input_output.dimension(0);
        int m = input_output.dimension(1);
        int l = input_output.dimension(2);

        fftw_complex *in = nullptr;
        int direction = fftDirection == FFT_FORWARD ? FFTW_FORWARD : FFTW_BACKWARD;

        in = reinterpret_cast<fftw_complex *>(input_output.data());

        fftw_plan plan = fftw_plan_dft_3d(l, m, n, in, in, direction, FFTW_ESTIMATE);

        fftw_execute(plan);

        // Scaling (fftw does not scale by default)
        if (direction == FFTW_BACKWARD)
        {
            input_output = input_output * Complex(1.0 / (n * m * l), 0);
        }

        fftw_destroy_plan(plan);
    };

}

#endif //FFTWFFT_HPP