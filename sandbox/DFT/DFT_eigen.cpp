# define M_PI           3.14159265358979323846  /* pi */

#include <unsupported/Eigen/SparseExtra>
#include <iostream>                      
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>

#include <complex>

using namespace std;

using comp=std::complex<double>;

// i imaginary constant
const comp If(0.0f, 1.0f);


comp f(comp c){
	comp r(std::abs(c)*std::abs(c),0);
	return r;
}

int main(int argc, char** argv)
{
  //using namespace LinearAlgebra;
	
	// -------------------
	// create the matrix W
	// -------------------
  int n = 4;
	//matrixXcd is just an alias for Matrix<std::complex<double>, Dynamic, Dynamic>
	Eigen::MatrixXcd W(n,n); 

	comp exponent = (2*M_PI/n)*If;
  for (int i=0; i<n; i++) {
		for(int j=0; j<n; j++){
			W(i,j)=std::exp(exponent*(1.0*i*j));
		}
  }

	// -------------------
	// create the vector x
	// -------------------
	Eigen::VectorXcd x(n);
	for(int i=0; i<n; i++){
		x(i)=f(i);
	}
	
	Eigen::VectorXcd X = W*x;	
	

	std::cout<<W<<std::endl;
	std::cout<<X<<std::endl;
	
	/*
  Eigen::saveMarketVector(b, "./DFT.mtx");
    
	FILE* out = fopen("DFT.mtx","w");
  fprintf(out,"%%%%MatrixMarket vector coordinate real general\n");
  fprintf(out,"%d\n", n);
  for (int i=0; i<n; i++) {
		fprintf(out,"%d %f\n", i ,X(i));
  }
  fclose(out);
  */
	return 0;
}
