int main(){
    FFTSolver fft_solver(std::make_unique<CUDAFFT>());
    ApplicationWrapper<double,3> w();
    //operazioni su w
    fft_solver.compute_fft_C2C(w,FFT_FORWARD);
}