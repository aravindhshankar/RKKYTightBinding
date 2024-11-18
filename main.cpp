#include<iostream> 
#include<Eigen/Dense>
#include<Eigen/Eigenvalues>
#include "include/sqstrain.h"
#define N 2000 

//using Eigen::MatrixXd, Eigen::Matrix2d; 
using namespace Eigen;
using namespace std::chrono;

//typedef Matrix <double,N,N> MatrixNd ;
//typedef Vector <std::complex<double>,N> VectorNcd ;

// int main()
// {
// 	std::cout << "Hello World!" << std::endl;

// 	MatrixXd mat(N,N); 
// 	VectorXcd eigvals(N); 
// 	mat << 1, 3, 
// 		   3, 1; 
//   	std::cout << mat << std::endl;
//   	Eigen::EigenSolver<MatrixXd> solver(mat);
//   	eigvals = solver.eigenvalues();
//   	std::cout << "Eigenvalues are : " << std::endl << 
//   					eigvals << std::endl;
//   	std::cout << "eigvals[0] = " << eigvals[0] << std::endl; 
//   	std::cout << "Output of testfunc : " << test_func(N) << std::endl;

//   	std::cout << "Returened to main : " << test_ret_mat(N) << std::endl;
//   	make_null(&mat);
//   	std::cout << "The matix m from the start is now : " << std::endl
//   											 << mat <<std::endl; 
// 	return (0); 
// }

int main ()
{
	MatrixXd mat(N,N);
	VectorXcd eigvals(N); 
	mat = make_ham(N); 
	float conv_to_Mb = 1024.0 * 1024.0 ; 
	auto start = high_resolution_clock::now();
	std::cout<< " Eigenvalue computation started " << std::endl;
	std::cout<< " Size of mat = " << sizeof(mat)/conv_to_Mb << std::endl;
	Eigen::EigenSolver<MatrixXd> solver(mat);
	eigvals = solver.eigenvalues();
	std::cout << "Eigenvalue computation of size " << N << " x "
						<< N << " completed." << std::endl;

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<seconds>(stop - start);
	std::cout << "Time taken : " << duration.count() << " seconds. " << std ::endl;
	return 0 ;

}
