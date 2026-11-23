import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

data = yf.download("SPY", start="2020-01-01", end="2025-01-01")["Adj Close"]
returns = np.log(data / data.shift(1)).dropna()

ar_model = ARIMA(returns, order=(1, 0, 0)).fit()

ma_model = ARIMA(returns, order=(0, 0, 1)).fit()

arima_model = ARIMA(returns, order=(1, 0, 1)).fit()

ar_forecast = ar_model.forecast(steps=30)
ma_forecast = ma_model.forecast(steps=30)
arima_forecast = arima_model.forecast(steps=30)

def calculate_rmse(model_forecast, actual_data):
    return np.sqrt(mean_squared_error(actual_data[-30:], model_forecast))

ar_rmse = calculate_rmse(ar_forecast, returns)
ma_rmse = calculate_rmse(ma_forecast, returns)
arima_rmse = calculate_rmse(arima_forecast, returns)

print(f"AR(1) RMSE: {ar_rmse:.4f}")
print(f"MA(1) RMSE: {ma_rmse:.4f}")

rmse_values = [ar_rmse, ma_rmse, arima_rmse]
model_names = ["AR(1)", "MA(1)", "ARIMA(1,0,1)"]

plt.figure(figsize=(8, 5))
plt.bar(model_names, rmse_values, color=["skyblue", "lightgreen", "salmon"])
plt.title("RMSE Comparison for AR(1), MA(1), ARIMA(1,0,1)")
plt.ylabel("RMSE")
plt.show()
print(f"ARIMA(1,0,1) RMSE: {arima_rmse:.4f}")
