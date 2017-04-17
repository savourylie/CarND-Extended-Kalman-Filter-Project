#include "kalman_filter.h"

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
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	// Get difference y between prediction and measurement
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;

	// Get Kalman gain
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

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
	double tan_phi = py / px;
	
	hx << sqrt(px*px + py*py), // rho
		atan(tan_phi), // phi
		(px * vx + py * vy) / sqrt(px*px + py*py); //  rho dot

	VectorXd y = z - hx;

	y = y.unaryExpr(&normalize_atan);

	// Get Kalman gain
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

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

