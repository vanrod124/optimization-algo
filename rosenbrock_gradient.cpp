#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd rosenbrock_gradient(const VectorXd &x) {
    VectorXd g(2);
    g(0) = -400 * x(0) * (x(1) - x(0) * x(0)) - 2 * (1 - x(0));
    g(1) = 200 * (x(1) - x(0) * x(0));
    return g;
}

double line_search(const VectorXd &x, const VectorXd &d, double alpha = 1.0, double beta = 0.5, double c = 0.0001) {
    double t = alpha;
    while ((x + t * d).transpose() * rosenbrock_gradient(x + t * d) < (x.transpose() * rosenbrock_gradient(x) + c * t * d.transpose() * rosenbrock_gradient(x)).transpose()) {
        t *= beta;
    }
    return t;
}

VectorXd bfgs(const VectorXd &initial_guess, const double epsilon = 1e-6, const int max_iterations = 1000) {
    VectorXd x = initial_guess;
    MatrixXd H = MatrixXd::Identity(2, 2);
    
    for (int k = 0; k < max_iterations; k++) {
        VectorXd g = rosenbrock_gradient(x);
        if (g.norm() < epsilon) {
            break;
        }

        VectorXd d = -H * g;
        double t = line_search(x, d);
        VectorXd x_new = x + t * d;
        VectorXd g_new = rosenbrock_gradient(x_new);
        VectorXd s = x_new - x;
        VectorXd y = g_new - g;
        double rho = 1.0 / y.dot(s);

        MatrixXd I = MatrixXd::Identity(2, 2);
        H = (I - rho * s * y.transpose()) * H * (I - rho * y * s.transpose()) + rho * s * s.transpose();
        x = x_new;
    }
    
    return x;
}

int main() {
    VectorXd initial_guess(2);
    initial_guess << -1.2, 1;
    VectorXd solution = bfgs(initial_guess);

    cout << "Solution: " << endl << solution << endl;
    return 0;
}
