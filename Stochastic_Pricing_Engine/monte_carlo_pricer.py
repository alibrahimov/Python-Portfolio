import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import logging

try:
    import mc_pricer_cpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    logging.warning("C++ extension 'mc_pricer_cpp' not found. Falling back to Python. Run 'python setup.py build_ext --inplace' to compile.")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MonteCarloPricingEngine:
    """
    A quantitative pricing engine that simulates asset paths using 
    Geometric Brownian Motion (GBM) and prices European options via 
    Monte Carlo simulations, comparing results to the Black-Scholes model.
    """
    
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float, 
                 simulations: int = 10000, steps: int = 252):
        self.S0 = S0          # Initial stock price
        self.K = K            # Strike price
        self.T = T            # Time to maturity (in years)
        self.r = r            # Risk-free interest rate
        self.sigma = sigma    # Volatility
        self.simulations = simulations
        self.steps = steps
        self.dt = T / steps
        self.price_paths = None
        
    def simulate_gbm_paths(self) -> np.ndarray:
        """Simulates asset price paths using Geometric Brownian Motion."""
        logging.info(f"Simulating {self.simulations} paths using GBM...")
        np.random.seed(42) # For reproducibility
        
        # Generate random standard normal variables
        Z = np.random.standard_normal((self.simulations, self.steps))
        
        # Brownian Motion (Wiener Process)
        W = np.cumsum(np.sqrt(self.dt) * Z, axis=1)
        W = np.column_stack([np.zeros(self.simulations), W])
        
        # Time array
        time = np.linspace(0, self.T, self.steps + 1)
        
        # Geometric Brownian Motion formula derived from Ito's Lemma
        drift = (self.r - 0.5 * self.sigma**2) * time
        diffusion = self.sigma * W
        
        self.price_paths = self.S0 * np.exp(drift + diffusion)
        return self.price_paths

    def black_scholes_call(self) -> float:
        """Calculates the theoretical European Call price using Black-Scholes."""
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        call_price = (self.S0 * norm.cdf(d1)) - (self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        return call_price

    def monte_carlo_call(self) -> float:
        def monte_carlo_call_cpp(self) -> float:
        """Prices the European Call option using the optimized C++ engine."""
        if not CPP_AVAILABLE:
            logging.error("C++ module not compiled. Using Python fallback.")
            return self.monte_carlo_call()
            
        logging.info(f"Routing {self.simulations} simulations through C++ engine...")
        mc_price = mc_pricer_cpp.monte_carlo_cpp(
            self.S0, self.K, self.T, self.r, self.sigma, self.simulations, self.steps
        )
        return mc_price
        """Prices the European Call option using Monte Carlo expected payoff."""
        if self.price_paths is None:
            self.simulate_gbm_paths()
            
        terminal_prices = self.price_paths[:, -1]
        
        # Option payoff at maturity: max(S_T - K, 0)
        payoffs = np.maximum(terminal_prices - self.K, 0)
        
        # Discount the expected payoff back to present value
        mc_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        return mc_price

    def plot_paths(self, num_paths_to_plot: int = 100) -> None:
        """Visualizes a subset of the simulated asset paths."""
        if self.price_paths is None:
            self.simulate_gbm_paths()
            
        time = np.linspace(0, self.T, self.steps + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(time, self.price_paths[:num_paths_to_plot].T, lw=1.5, alpha=0.6)
        plt.axhline(y=self.K, color='r', linestyle='--', label=f"Strike Price (K={self.K})")
        plt.title(f"Geometric Brownian Motion: {num_paths_to_plot} Simulated Paths")
        plt.xlabel("Time (Years)")
        plt.ylabel("Asset Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # Parameters for the pricing engine
    engine = MonteCarloPricingEngine(
        S0=100.0,    # Current Price
        K=100.0,     # Strike Price (At-the-money)
        T=1.0,       # 1 Year to expiry
        r=0.05,      # 5% risk-free rate
        sigma=0.2,   # 20% volatility
        simulations=50000, 
        steps=252
    )
    
    # 1. Run the simulation
    engine.simulate_gbm_paths()
    
    # 2. Compare Pricing Models
    bs_price = engine.black_scholes_call()
    mc_price = engine.monte_carlo_call()
    
    print("\n--- Options Pricing Results ---")
    print(f"Analytical Black-Scholes Price: ${bs_price:.4f}")
    print(f"Numerical Monte Carlo Price:    ${mc_price:.4f}")
    print(f"Pricing Error:                  ${abs(bs_price - mc_price):.6f}")
    
    # 3. Visualize
    engine.plot_paths(num_paths_to_plot=100)
