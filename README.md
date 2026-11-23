# Ali Ibrahimov – Quantitative Finance & Data Projects

Finance analyst with an MSc in Finance with Data Science from UCL. I work in FP&A at Canonical and use Python for risk, forecasting and quantitative modelling. This repo collects projects I have built along the way.

Looking for: data analyst, risk analyst, quant analyst and finance analyst roles.

## Projects

### Risk and portfolio construction

| Project | What it does | Methods |
| :--- | :--- | :--- |
| [Portfolio Optimizer](Portfolio_Optimizer) | Builds a long-only mean-variance efficient frontier for four large-cap stocks and estimates the portfolio's 95% VaR by Monte Carlo simulation. | Mean-variance optimisation (cvxpy), efficient frontier, Monte Carlo VaR |
| [Factor Model](Factor_Model) | Estimates CAPM alphas and betas against SPY, compares them with a multi-regressor model and checks for multicollinearity. | OLS regression, CAPM, VIF |
| [AR-GARCH Volatility Targeting](AR_GARCH_Volatility_Targeting) | Uses AR(1) for the direction signal and GARCH(1,1) to scale position size to a volatility target. Backtested on SPY, 2010–2025. | ARIMA, GARCH, volatility targeting |

### Forecasting and machine learning

| Project | What it does | Methods |
| :--- | :--- | :--- |
| [Time Series Forecasting](Time_Series_Forecasting) | Compares AR(1), MA(1) and ARIMA(1,0,1) forecasts of SPY returns by RMSE, plus a PCA of large-cap tech returns. | ARIMA, RMSE, PCA |
| [ML Alpha Strategy](ML_Alpha_Walk_Forward) | Predicts next-day direction with logistic regression and a random forest using walk-forward validation, then compares CAGR, Sharpe and max drawdown. | Classification, walk-forward validation |
| [Cross-Sectional ML Factor Model](Cross_Sectional_ML_Arbitrage) | Predicts 21-day forward returns from momentum, volatility and mean-reversion features (cross-sectional z-scores) and reports Sharpe, drawdown and information coefficient. | Random forest, feature engineering, IC |

### Trading research

| Project | What it does | Methods |
| :--- | :--- | :--- |
| [Pairs Trading](Pairs_Trading_Stat_Arb) | Tests KO/PEP for cointegration and trades the z-score of the hedged spread. | Engle-Granger cointegration, OLS hedge ratio |
| [Intraday Strategy Backtest](Trading_Strategy_Backtest) | Backtests moving-average crossover and mean-reversion signals on 5-minute SPY bars, including transaction costs and slippage. | Sharpe, max drawdown, hit rate, turnover |

### Derivatives and fixed income

| Project | What it does | Methods |
| :--- | :--- | :--- |
| [Stochastic Pricing Engine](Stochastic_Pricing_Engine) | Prices European calls by Monte Carlo under GBM and checks the result against Black-Scholes. The simulation loop has a C++ version (pybind11), and a unit test runs on every push through GitHub Actions. | GBM, Monte Carlo, Black-Scholes, C++ |
| [Stochastic Calculus](Stochastic_Calculus) | Simulates Brownian motion and GBM (exact solution vs Euler-Maruyama) and illustrates Itô's lemma numerically. | Itô calculus, Euler-Maruyama |
| [Yield Curve Analysis](Yield_Curve_Analysis) | Bond pricing, YTM, duration, convexity and DV01; bootstraps spot rates and fits a Nelson-Siegel curve. | Bootstrapping, Nelson-Siegel |

Earlier exercises (statistical tests, plotting, Python basics) are in [Learning](Learning).

## Running the code

    git clone https://github.com/alibrahimov/Python-Portfolio.git
    cd Python-Portfolio
    pip install -r requirements.txt
    python Factor_Model/factor_model.py

Market data comes from Yahoo Finance through yfinance, so results change with the download date.

## Contact

- LinkedIn: https://www.linkedin.com/in/ali-ibrahimov/
- Email: ali.ibrahimov4@gmail.com
