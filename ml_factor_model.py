import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLFactorStrategy:
    """
    A cross-sectional machine learning factor strategy pipeline.
    Ingests market data, calculates alpha factors, trains a Random Forest model,
    and backtests a long/short decile portfolio.
    """

    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.DataFrame()
        self.features = pd.DataFrame()
        self.model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        
    def fetch_data(self) -> None:
        """Downloads adjusted close prices from Yahoo Finance."""
        logging.info(f"Fetching data for {len(self.tickers)} tickers from {self.start_date} to {self.end_date}")
        raw_data = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)
        self.data = raw_data["Close"]
        
    def engineer_features(self) -> None:
        """Calculates momentum, volatility, and mean reversion factors, applying cross-sectional Z-scoring."""
        logging.info("Engineering factors and target variables...")
        returns = np.log(self.data).diff()
        
        # Factor generation
        momentum = self.data.pct_change(21)
        volatility = returns.rolling(21).std()
        mean_reversion = -returns.rolling(5).sum()
        
        # Target variable (21-day forward returns)
        forward_returns = returns.shift(-21)
        
        # Combine into multi-index DataFrame
        features = pd.concat([momentum.stack(), volatility.stack(), mean_reversion.stack()], axis=1)
        features.columns = ["momentum", "volatility", "mean_reversion"]
        features["forward_return"] = forward_returns.stack()
        
        features = features.dropna()
        
        # Cross-sectional Z-score per day
        def cross_sectional_zscore(df):
            return (df - df.mean()) / df.std()
            
        cols_to_zscore = ["momentum", "volatility", "mean_reversion"]
        features[cols_to_zscore] = features.groupby(level=0)[cols_to_zscore].transform(cross_sectional_zscore)
        
        self.features = features
        logging.info("Feature engineering complete.")

    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
        """Splits the dataset into 70% training and 30% testing temporally."""
        dates = self.features.index.get_level_values(0).unique()
        train_cutoff = int(len(dates) * 0.7)
        
        train_dates = dates[:train_cutoff]
        test_dates = dates[train_cutoff:]
        
        train_data = self.features.loc[train_dates]
        test_data = self.features.loc[test_dates]
        
        return train_data, test_data, test_dates

    def run_backtest(self) -> pd.DataFrame:
        """Trains the model and executes the historical backtest."""
        train_data, test_data, test_dates = self._split_data()
        feature_cols = ["momentum", "volatility", "mean_reversion"]
        
        # Train Model
        logging.info("Training Random Forest model on historical data...")
        X_train = train_data[feature_cols]
        y_train = train_data["forward_return"]
        self.model.fit(X_train, y_train)
        
        # Generate Predictions
        logging.info("Generating out-of-sample predictions...")
        X_test = test_data[feature_cols]
        preds = self.model.predict(X_test)
        
        predictions = pd.Series(preds, index=X_test.index)
        
        # Portfolio Construction
        logging.info("Constructing top/bottom decile portfolios...")
        portfolio_returns = []
        ic_values = []
        
        for date in test_dates:
            daily_pred = predictions.loc[date]
            daily_true = self.features.loc[date]["forward_return"]
            
            # Information Coefficient (IC)
            ic_values.append(daily_pred.corr(daily_true))
            
            # Long/Short Construction
            n = len(daily_pred)
            if n > 0:
                top_decile = daily_pred.nlargest(max(1, int(n * 0.1))).index
                bottom_decile = daily_pred.nsmallest(max(1, int(n * 0.1))).index
                
                long_ret = daily_true.loc[top_decile].mean()
                short_ret = daily_true.loc[bottom_decile].mean()
                portfolio_returns.append(long_ret - short_ret)
            else:
                portfolio_returns.append(0)
                
        self.portfolio_returns = pd.Series(portfolio_returns, index=test_dates)
        self.ic_series = pd.Series(ic_values, index=test_dates)
        
        return self._calculate_metrics()

    def _calculate_metrics(self) -> pd.DataFrame:
        """Calculates standard quantitative performance metrics."""
        cum_returns = (1 + self.portfolio_returns).cumprod()
        
        cagr = cum_returns.iloc[-1] ** (12 / len(self.portfolio_returns)) - 1
        vol = self.portfolio_returns.std() * np.sqrt(12)
        sharpe = cagr / vol if vol != 0 else 0
        
        drawdown = cum_returns / cum_returns.cummax() - 1
        max_dd = drawdown.min()
        mean_ic = self.ic_series.mean()
        
        results = pd.DataFrame({
            "Metric": ["CAGR", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown", "Mean IC"],
            "Value": [round(cagr, 4), round(vol, 4), round(sharpe, 4), round(max_dd, 4), round(mean_ic, 4)]
        })
        
        self.cum_returns = cum_returns
        return results

    def plot_equity_curve(self) -> None:
        """Visualizes the strategy's cumulative returns."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cum_returns, label="Long/Short Portfolio")
        plt.title("Cross-Sectional ML Factor Strategy Equity Curve")
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
    universe = [
        "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","JPM","V","UNH",
        "HD","PG","MA","DIS","BAC","XOM","KO","PFE","INTC","CSCO",
        "CMCSA","PEP","ABT","T","CRM","ADBE","NFLX","WMT","COST","ORCL",
        "AVGO","ACN","MCD","DHR","QCOM","LLY","MDT","TXN","HON","LIN",
        "LOW","AMGN","IBM","INTU","NEE","UPS","PM","RTX","SPGI","BA"
    ]
    
    strategy = MLFactorStrategy(tickers=universe, start_date="2012-01-01", end_date="2025-01-01")
    strategy.fetch_data()
    strategy.engineer_features()
    
    performance_metrics = strategy.run_backtest()
    print("\n--- Strategy Performance ---")
    print(performance_metrics.to_string(index=False))
    
    strategy.plot_equity_curve()
