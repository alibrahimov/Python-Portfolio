import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt
import logging
import warnings

# Suppress statistical convergence warnings for clean terminal output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ARGarchVolatilityTargeting:
    """
    A time-series trading strategy that uses an AR(1) model for directional 
    signals and a GARCH(1,1) model for dynamic volatility targeting.
    """

    def __init__(self, ticker: str = "SPY", start_date: str = "2010-01-01", end_date: str = "2025-01-01",
                 train_window: int = 756, rebalance_freq: int = 21, target_vol: float = 0.10):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.train_window = train_window
        self.rebalance_freq = rebalance_freq
        self.target_vol = target_vol
        
        self.prices = pd.Series(dtype=float)
        self.returns = pd.Series(dtype=float)
        self.strategy_returns = pd.Series(dtype=float)
        
    def fetch_data(self) -> None:
        """Downloads adjusted close prices and calculates log returns."""
        logging.info(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}")
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)
        self.prices = data["Close"]
        self.returns = np.log(self.prices).diff().dropna()

    def run_backtest(self) -> pd.DataFrame:
        """Executes the rolling AR-GARCH backtest."""
        logging.info(f"Starting rolling backtest (Window: {self.train_window}, Step: {self.rebalance_freq}). This may take a moment...")
        
        signals = []
        realized_returns = []
        
        # Iterate through the time series
        for t in range(self.train_window, len(self.returns), self.rebalance_freq):
            train_returns = self.returns.iloc[t - self.train_window : t]
            
            # 1. Directional Signal: AR(1) Model
            ar_model = ARIMA(train_returns, order=(1, 0, 0)).fit()
            ar_forecast = ar_model.forecast(steps=self.rebalance_freq)
            
            # 2. Risk Management: GARCH(1,1) Model (Scaled by 100 for optimizer convergence)
            garch = arch_model(train_returns * 100, vol="Garch", p=1, q=1)
            garch_fit = garch.fit(disp="off")
            # Unscale the forecasted variance
            vol_forecast = np.sqrt(garch_fit.forecast(horizon=self.rebalance_freq).variance.values[-1]) / 100
            
            # 3. Position Sizing
            position = np.sign(ar_forecast)
            scaled_position = position * (self.target_vol / vol_forecast)
            
            # 4. Forward Returns Calculation
            future_returns = self.returns.iloc[t : t + self.rebalance_freq]
            strat_returns = scaled_position[:len(future_returns)] * future_returns.values
            
            signals.extend(scaled_position[:len(future_returns)])
            realized_returns.extend(strat_returns)
            
            # Log progress every ~5 years of simulated data
            if t % (252 * 5) < self.rebalance_freq:
                logging.info(f"Backtest progress: {self.returns.index[t].date()} reached.")

        # Align strategy returns with actual dates
        index_start = self.train_window
        index_end = self.train_window + len(realized_returns)
        self.strategy_returns = pd.Series(realized_returns, index=self.returns.index[index_start:index_end])
        
        logging.info("Backtest complete.")
        return self._calculate_metrics()

    def _calculate_metrics(self) -> pd.DataFrame:
        """Calculates standard performance metrics."""
        cum_returns = (1 + self.strategy_returns).cumprod()
        
        cagr = cum_returns.iloc[-1] ** (252 / len(self.strategy_returns)) - 1
        vol = self.strategy_returns.std() * np.sqrt(252)
        sharpe = cagr / vol if vol != 0 else 0
        
        drawdown = cum_returns / cum_returns.cummax() - 1
        max_dd = drawdown.min()
        
        self.cum_returns = cum_returns
        
        results = pd.DataFrame({
            "Metric": ["CAGR", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown"],
            "Value": [round(cagr, 4), round(vol, 4), round(sharpe, 4), round(max_dd, 4)]
        })
        return results

    def plot_equity_curve(self) -> None:
        """Visualizes the strategy's cumulative returns."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cum_returns, label="AR-GARCH Volatility Targeted Strategy", color='blue')
        plt.title(f"Equity Curve: AR-GARCH Volatility Targeting ({self.ticker})")
        plt.ylabel("Cumulative Return")
        plt.xlabel("Date")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    strategy = ARGarchVolatilityTargeting(
        ticker="SPY", 
        start_date="2010-01-01", 
        end_date="2025-01-01",
        train_window=756,      # 3 years of trading days
        rebalance_freq=21,     # 1 trading month
        target_vol=0.10        # Target 10% annualized volatility
    )
    
    strategy.fetch_data()
    performance = strategy.run_backtest()
    
    print("\n--- Strategy Performance ---")
    print(performance.to_string(index=False))
    
    strategy.plot_equity_curve()
