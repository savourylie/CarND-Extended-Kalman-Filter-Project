#include "kalman_filter.h"
#include <iostream>

using std::cout;
using Eigen::MatrixXd;
using Eigen::VectorXd;

double normalize_atan(double theta);

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
	x_ = F_ * x_; // F: 4x4, x: 4x1
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_; // P, Q: 4x4
}

void KalmanFilter::Update(const VectorXd &z) {
	// Get difference y between prediction and measurement
	VectorXd z_pred = H_ * x_; // H: 2x4, x: 4x1, z_pred: 2x1
	VectorXd y = z - z_pred; // z: 2x1, y: 2x1

	// Get Kalman gain
	MatrixXd Ht = H_.transpose(); // Ht: 4x2
	MatrixXd S = H_ * P_ * Ht + R_; // S, R_laser: 2x2, P: 4x4
	MatrixXd Si = S.inverse(); // Si: 2x2
	MatrixXd PHt = P_ * Ht; // PHt: 4x2
	MatrixXd K = PHt * Si; // K: 4x2

	// Update x_ and P_ using Kalman gain
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	// Get difference y between prediction and measurement
	VectorXd hx = VectorXd(3);
	double px = x_(0);
	double py = x_(1);
	double vx = x_(2);
	double vy = x_(3);
	
	hx << sqrt(px*px + py*py), // rho
		atan2(py, px), // phi
		(px * vx + py * vy) / sqrt(px*px + py*py); //  rho dot

	VectorXd y = z - hx; // z: 3x1, hx: 3x1, y: 3x1

	y(1) = normalize_atan(y(1)); // Normalize y so that it's between -pi and pi

	// Get Kalman gain
	MatrixXd Ht = H_.transpose(); // H: 3x4, Ht: 4x3
	MatrixXd S = H_ * P_ * Ht + R_; // P: 4x4, S: 3x3
	MatrixXd Si = S.inverse(); // Si: 3x3
	MatrixXd PHt = P_ * Ht; // PHt: 4x3
	MatrixXd K = PHt * Si; // K: 4x3

	// Update x_ and P_ using Kalman gain
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

double normalize_atan(double theta) {
	// Normalize tan_phi so it's between -pi and pi
	while (theta > M_PI || theta <= -M_PI) {
		if (theta > M_PI) {
			theta -= 2*M_PI;
		}

		if (theta <= -M_PI) {
			theta += 2*M_PI;
		}
	}

	return theta;
}

