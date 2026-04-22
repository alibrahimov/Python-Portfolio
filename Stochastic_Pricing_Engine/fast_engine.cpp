#include <pybind11/pybind11.h>
#include <cmath>
#include <random>
#include <algorithm>

namespace py = pybind11;

// The heavy C++ simulation loop
double monte_carlo_cpp(double S0, double K, double T, double r, double sigma, int simulations, int steps) {
    std::mt19937 generator(42); 
    std::normal_distribution<double> distribution(0.0, 1.0);
    
    double dt = T / steps;
    double payoff_sum = 0.0;

    for (int i = 0; i < simulations; ++i) {
        double S = S0;
        for (int j = 0; j < steps; ++j) {
            double Z = distribution(generator);
            S *= exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z);
        }
        payoff_sum += std::max(S - K, 0.0);
    }
    return exp(-r * T) * (payoff_sum / simulations);
}

PYBIND11_MODULE(mc_pricer_cpp, m) {
    m.def("monte_carlo_cpp", &monte_carlo_cpp, "Monte Carlo options pricing in C++");
}
