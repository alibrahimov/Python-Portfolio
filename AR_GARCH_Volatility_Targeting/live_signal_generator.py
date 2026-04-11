import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
import logging
from datetime import datetime

# Suppress statistical convergence warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_live_signal(ticker: str = "SPY", train_window: int = 756, target_vol: float = 0.10):
    """
    Fetches live market data and generates the Out-Of-Sample (OOS) 
    position sizing for the current trading day.
    """
    logging.info(f"Fetching latest market data for {ticker}...")
    
    # Fetch 4 years of data to ensure we have enough trading days for the 756-day window
    data = yf.download(ticker, period="4y", auto_adjust=True, progress=False)
    
    if data.empty:
        logging.error("Failed to fetch data from Yahoo Finance. Check your internet connection.")
        return
        
    returns = np.log(data["Close"]).diff().dropna()
    
    # Isolate the exact 3-year rolling window up to TODAY
    recent_returns = returns.iloc[-train_window:]
    latest_date = recent_returns.index[-1].date()
    logging.info(f"Latest market close captured: {latest_date}")
    
    # 1. Directional Signal: AR(1) Model
    logging.info("Fitting AR(1) model for directional forecast...")
    ar_model = ARIMA(recent_returns, order=(1, 0, 0)).fit()
    ar_forecast = ar_model.forecast(steps=1).iloc[0]  # Forecast exactly 1 day ahead
    direction = np.sign(ar_forecast)
    
    # 2. Risk Management: GARCH(1,1) Model
    logging.info("Fitting GARCH(1,1) for volatility forecast...")
    garch = arch_model(recent_returns * 100, vol="Garch", p=1, q=1)
    garch_fit = garch.fit(disp="off")
    
    # Forecast next day variance, unscale it
    vol_forecast = np.sqrt(garch_fit.forecast(horizon=1).variance.values[-1, 0]) / 100
    
    # 3. Position Sizing
    # If volatility is high, exposure drops. If volatility is low, exposure increases.
    position_size = direction * (target_vol / vol_forecast)
    
    # 4. Professional Output
    print("\n" + "="*55)
    print(f" LIVE OOS TRADING SIGNAL FOR {ticker}")
    print(f" Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)
    print(f" Directional Forecast (AR):   {'LONG' if direction > 0 else 'SHORT'} (Raw Alpha: {ar_forecast:.6f})")
    print(f" Forecasted Volatility:       {vol_forecast * 100:.2f}% annualized")
    print(f" Target Position Size:        {position_size * 100:.2f}% of Total Capital")
    print("="*55 + "\n")

if __name__ == "__main__":
    generate_live_signal(ticker="SPY", train_window=756, target_vol=0.10)
