namespace fftcore{

	class FFTEngine{
		public:
			void virtual fft(const& TensorFFT, &TensorFFT) const = 0;
			void virtual ifft(const& TensorFFT, &TensorFFT) const = 0;
			void virtual fft(&TensorFFT) const = 0;
			void virtual ifft(&TensorFFT) const = 0;
			
			virtual ~FFTEngine() = default; // da rivedere
	}
}

