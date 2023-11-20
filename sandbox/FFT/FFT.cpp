# define M_PI           3.14159265358979323846  /* pi */

#include <complex>
#include <vector>
#define N 16


using comp=std::complex<double>;


// i imaginary constant
const comp If(0.0f, 1.0f);


std::vector<comp> FFT(std::vector<comp> &x){
	std::vector<comp> y(x.size());
	int n = x.size();
	if(n==1){
		y[0]=x[0];
		return y;
	}

	comp w_root = std::exp(-(2*M_PI/n)*If);
	comp w = 1;
	
	//split the vector in two parts
	std::vector<comp> x_e; //x_even
	std::vector<comp> x_o; //x_odd
	
	for(int i=0; i<n;i++){
		if(i%2==0) x_e.push_back(x[i]);
		else x_o.push_back(x[i]);
	}

	std::vector<comp> y_e = FFT(x_e);
	std::vector<comp> y_o = FFT(x_o);

	for(int j=0; j<n/2; j++){
		y[j] =  y_e[j]+w*y_o[j];
		y[j+n/2] = y_e[j]-w*y_o[j];
		w = w*w_root;
	}	

	return y;
	
}

comp f(comp &c){
	comp r(std::abs(c)*std::abs(c),0);
	return r;
}

int main(int argc, char** argv)
{
  int n = 16;

	// -------------------
	// create the vector x
	// -------------------
	std::vector<comp> y(n);
	std::vector<comp> x(n);
	for(int i=0; i<n; i++){
		comp c(i,0);
		x[i]=f(c);
	}
	y=FFT(x);
	
	for(int i =0;i<n; i++){
		printf("%f+%fi\t", y[i].real(), y[i].imag());
	}	
	printf("\n");

	return 0;
}
