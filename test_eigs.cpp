#include<iostream> 
#include<Eigen/Dense>
#include<Eigen/Eigenvalues>
#define N 2 

//using Eigen::MatrixXd, Eigen::Matrix2d; 
using namespace Eigen ;

int main()
{
	std::cout << "Hello World!" << std::endl;
	typedef Matrix <double,N,N> MatrixNd ;
	typedef Vector <std::complex<double>,N> VectorNcd ;
	MatrixNd m;
	VectorNcd eigvals; 
	m << 1, 3 , 
		 3, 1 ;
  	std::cout << m << std::endl;
  	Eigen::EigenSolver<MatrixNd> solver(m);
  	eigvals = solver.eigenvalues();
  	std::cout << "Eigenvalues are : " << std::endl << 
  					eigvals << std::endl;
  	std::cout << "eigvals[0] = " << eigvals[0] << std::endl; 

	return (0); 
}