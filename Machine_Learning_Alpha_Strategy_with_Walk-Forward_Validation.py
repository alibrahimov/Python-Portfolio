import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

ticker = "SPY"
start = "2010-01-01"
end = "2025-01-01"

data = yf.download(ticker, start=start, end=end, auto_adjust=True)
data["Return"] = np.log(data["Close"]).diff()
data["Volatility"] = data["Return"].rolling(20).std()
data["Momentum"] = data["Close"].pct_change(20)
data["Volume_Change"] = data["Volume"].pct_change()
data["Lag1"] = data["Return"].shift(1)
data["Lag2"] = data["Return"].shift(2)
data["Lag3"] = data["Return"].shift(3)

data["Forward_Return"] = data["Return"].shift(-1)
data["Label"] = (data["Forward_Return"] > 0).astype(int)

data = data.dropna()

features = ["Volatility", "Momentum", "Volume_Change", "Lag1", "Lag2", "Lag3"]
X = data[features]
y = data["Label"]

train_size = int(len(data) * 0.7)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_probs = log_model.predict_proba(X_test_scaled)[:, 1]
log_pred = log_model.predict(X_test_scaled)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=5)
rf_model.fit(X_train, y_train)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_pred = rf_model.predict(X_test)

log_accuracy = accuracy_score(y_test, log_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

positions_log = (log_probs - 0.5) * 2
positions_rf = (rf_probs - 0.5) * 2

test_returns = data["Return"].iloc[train_size:]

strategy_log = positions_log * test_returns.values
strategy_rf = positions_rf * test_returns.values

cum_log = (1 + strategy_log).cumprod()
cum_rf = (1 + strategy_rf).cumprod()

def performance_metrics(strategy_returns):
    cumulative = (1 + strategy_returns).cumprod()
    cagr = cumulative.iloc[-1] ** (252 / len(strategy_returns)) - 1
    vol = strategy_returns.std() * np.sqrt(252)
    sharpe = cagr / vol
    drawdown = cumulative / cumulative.cummax() - 1
    max_dd = drawdown.min()
    return cagr, vol, sharpe, max_dd

log_cagr, log_vol, log_sharpe, log_dd = performance_metrics(pd.Series(strategy_log))
rf_cagr, rf_vol, rf_sharpe, rf_dd = performance_metrics(pd.Series(strategy_rf))

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [log_accuracy, rf_accuracy],
    "CAGR": [log_cagr, rf_cagr],
    "Volatility": [log_vol, rf_vol],
    "Sharpe Ratio": [log_sharpe, rf_sharpe],
    "Max Drawdown": [log_dd, rf_dd]
})

print(results)

plt.figure(figsize=(10,6))
plt.plot(cum_log, label="Logistic Strategy")
plt.plot(cum_rf, label="Random Forest Strategy")
plt.legend()
plt.title("Equity Curve")
plt.show()
