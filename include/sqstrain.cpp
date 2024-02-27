#include <Eigen/Dense>
#include "sqstrain.h"
#include<iostream>
using Eigen::MatrixXd;

int test_func(unsigned int N)
{	
	MatrixXd m(N,N) ; 
	m << 1,3,2,3 ; 
	std::cout << m << std::endl;
	return 69; 
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
    for(size_t i =0 ; i<matrixPtr->rows(); i++)
    	for (size_t j=0 ; j< matrixPtr->cols(); j++) 
    		(*matrixPtr) (i,j) = 0;
    //matrixPtr->setZero();
}



