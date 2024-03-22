#include <Eigen/Dense>
#include "sqstrain.h"
#include<iostream>
using Eigen::MatrixXd;

int test_func(unsigned int N)
{	
	MatrixXd m(N,N) ; 
	int num = 5; 
	m << 1,3,2,3 ; 
	std::cout << m << std::endl;
	return num; 
}

/* Need to check if objects are return by reference by default 
i.e. is the returned variable an object or a pointer to an object? */

MatrixXd test_ret_mat(unsigned int N)
{
	MatrixXd m(N,N) ; 
	m << 1,2,3,4 ; 
	std::cout << " Now in test_ret_mat" << std::endl; 
	return m ;
}

void make_null(Eigen::MatrixXd* matrixPtr)
{
	if (!matrixPtr) {
        std::cerr << "Invalid pointer!" << std::endl;
        return;
    }
    for(auto i = 0 ; i<matrixPtr->rows(); i++)
    	for (auto j=0 ; j< matrixPtr->cols(); j++) 
    		(*matrixPtr) (i,j) = 0;
    //matrixPtr->setZero();
}


MatrixXd make_ham(int size)
{
	MatrixXd ham(size, size); 
	ham.setRandom(size,size);
	return ham;
}

