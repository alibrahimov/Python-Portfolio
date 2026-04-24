import unittest
from monte_carlo_pricer import MonteCarloPricingEngine

class TestOptionsPricing(unittest.TestCase):
    def test_black_scholes_call(self):
        """
        Tests the analytical Black-Scholes pricing formula against a known mathematical constant.
        Parameters: S0=100, K=100, T=1.0, r=0.05, sigma=0.2
        Expected Call Price: ~10.4506
        """
        engine = MonteCarloPricingEngine(
            S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, simulations=100, steps=252
        )
        
        calculated_price = engine.black_scholes_call()
        
        # We assert that the calculated price is almost equal to the true mathematical price (up to 3 decimal places)
        self.assertAlmostEqual(calculated_price, 10.4506, places=3, msg="Black-Scholes math is broken!")

if __name__ == '__main__':
    unittest.main()
