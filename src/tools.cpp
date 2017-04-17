#include <iostream>
#include "tools.h"
#include <stdexcept>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	// Calculate RMSE
	// Check vector dimensions
	int est_dim = estimations.size();
	int truth_dim = ground_truth.size();

	if (est_dim != truth_dim || est_dim == 0) {
		throw std::invalid_argument( "Vector dimensions don't match, or incorrect estimation dimension");
	}

	VectorXd rmse = VectorXd::Zero(estimations[0].size());
	
	// Accumulate squared residuals
	VectorXd res;
	VectorXd sq_res;

	for (int i = 0; i < est_dim; ++i ){
	    res = estimations[i] - ground_truth[i];
	    sq_res = res.array() * res.array();
	    rmse = rmse + sq_res;
//		cout << res;
	}

	// Calculate the mean
	
	rmse = rmse / est_dim;
	
	// Calculate the squared root
    rmse = rmse.array().sqrt();
	
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	MatrixXd Hj(3, 4);
	// Recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	// Check division by zero
	if (px == 0 && py == 0) {
	    throw std::invalid_argument("CalculateJacobian - Error - Division by Zero");    
	}
	
	// Compute the Jacobian matrix
    Hj << px / sqrt(px*px + py*py), py / sqrt(px*px + py*py), 0, 0,
        -py / (px*px + py*py), px / (px*px + py*py), 0, 0,
        py * (vx * py - vy * px) / pow(sqrt(px*px + py*py), 3), px * (vy * px - vx * py) / pow(sqrt(px*px + py*py), 3), px / sqrt(px*px + py*py), py / sqrt(px*px + py*py);
    
	return Hj;
}
