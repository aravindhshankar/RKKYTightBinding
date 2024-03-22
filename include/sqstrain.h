#ifndef SQSTRAIN_H 
#define SQSTRAIN_H


#include<Eigen/Dense>
//using namesapce Eigen
using Eigen::MatrixXd;

int test_func(unsigned int); 
MatrixXd test_ret_mat (unsigned int);
void make_null(MatrixXd*);
MatrixXd make_ham(int size);

#endif