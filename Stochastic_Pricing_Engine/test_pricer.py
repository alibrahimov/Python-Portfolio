import unittest
from monte_carlo_pricer import MonteCarloPricingEngine

class TestOptionsPricing(unittest.TestCase):
    def test_black_scholes_call(self):
        engine = MonteCarloPricingEngine(
            S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, simulations=100, steps=252
        )
        
        calculated_price = engine.black_scholes_call()
        
        self.assertAlmostEqual(calculated_price, 10.4506, places=3, msg="Black-Scholes math is broken!")

if __name__ == '__main__':
    unittest.main()
