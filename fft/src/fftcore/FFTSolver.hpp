namespace fftcore{
	class FFTSolver{
		public:
			void compute_fft(const& TensorFFTBase,&TensorFFTBase) const; // virtual?
			void ifft(const& TensorFFTBase,&TensorFFTBase) const;
			void fft(&TensorFFTBase) const;
			void ifft(&TensorFFTBase) const;
			void set_strategy(&&std::unique_ptr<FFTEngine>);
		private:
			std::unique_ptr<FFTEngine> _fftengine;
	}
}
