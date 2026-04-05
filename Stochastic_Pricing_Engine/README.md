# Stochastic Pricing Engine: Monte Carlo & Black-Scholes

## Overview
This project implements an object-oriented derivatives pricing engine in Python. It simulates asset price trajectories using Geometric Brownian Motion (GBM) and uses these stochastic paths to price European Call options via Monte Carlo methods. The numerical results are then benchmarked against the analytical Black-Scholes closed-form solution.

## Mathematical Foundation
The engine relies on the core principles of stochastic calculus and Ito's Lemma.

### 1. Asset Price Dynamics (GBM)
The asset price $S_t$ is assumed to follow a Stochastic Differential Equation (SDE) known as Geometric Brownian Motion:
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$
Where $dW_t$ is a standard Wiener process (Brownian motion). By applying Ito's Lemma, we solve for the continuous-time price path:
$$S_t = S_0 \exp\left( \left( \mu - \frac{\sigma^2}{2} \right)t + \sigma W_t \right)$$

### 2. Monte Carlo Option Pricing
According to risk-neutral valuation, the price of a derivative is the discounted expected value of its future payoff. For a European Call option with strike $K$:
$$C = e^{-rT} \mathbb{E}^{\mathbb{Q}}[\max(S_T - K, 0)]$$
The engine simulates 50,000 independent price paths to converge on this expected value using the Law of Large Numbers.

## Implementation Details
* **Algorithm:** Euler-Maruyama discretization for path generation.
* **Variance Reduction:** Standard `numpy` vectorization is utilized for heavy matrix operations, avoiding slow iterative loops.
* **Validation:** The Monte Carlo output is dynamically verified against the precise Black-Scholes equation, typically achieving a pricing error margin of less than $0.05.

## How to Run
```bash
pip install numpy pandas matplotlib scipy
python monte_carlo_pricer.py
