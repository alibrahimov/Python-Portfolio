import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def download_intraday(ticker="SPY", period="60d", interval="5m"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1)
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df

def infer_bars_per_day(idx):
    if len(idx) < 3:
        return 78
    dt = (idx[2] - idx[1])
    if not isinstance(dt, pd.Timedelta) or dt <= pd.Timedelta(0):
        return 78
    minutes = dt.total_seconds() / 60.0
    if minutes <= 0:
        return 78
    bars = int(round(390.0 / minutes))
    return max(bars, 1)

def bars_per_year(idx):
    bpd = infer_bars_per_day(idx)
    return 252 * bpd

def log_returns_from_close(close):
    return np.log(close / close.shift(1)).dropna()

def spread_proxy(df):
    hl = (df["High"] - df["Low"]).abs()
    mid = df["Close"].replace(0, np.nan)
    sp = (hl / mid).replace([np.inf, -np.inf], np.nan).dropna()
    return sp

def rolling_volatility(returns, window=78):
    return returns.rolling(window).std()

def equity_from_strategy(returns, strat_rets):
    eq = (1.0 + strat_rets.fillna(0.0)).cumprod()
    bh = (1.0 + returns.fillna(0.0)).cumprod()
    return eq, bh

def max_drawdown(equity):
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd.min()

def sharpe(returns, ann_factor):
    r = returns.dropna()
    if r.std(ddof=1) == 0 or len(r) < 3:
        return np.nan
    return (r.mean() / r.std(ddof=1)) * np.sqrt(ann_factor)

def hit_rate(returns):
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    return (r > 0).mean()

def turnover(position):
    p = position.fillna(0.0)
    return (p.diff().abs().fillna(0.0)).sum()

def apply_costs(position, returns, cost_per_turn=0.0002, slippage_per_turn=0.0001):
    pos = position.fillna(0.0)
    turns = pos.diff().abs().fillna(0.0)
    cost = turns * (cost_per_turn + slippage_per_turn)
    strat = pos.shift(1).fillna(0.0) * returns - cost
    return strat

def ma_crossover_signals(close, fast=20, slow=50):
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    sig = (ma_fast > ma_slow).astype(float)
    pos = sig.replace(0.0, 0.0)
    return pos

def mean_reversion_signals(close, lookback=78, entry=1.0, exit=0.2):
    m = close.rolling(lookback).mean()
    s = close.rolling(lookback).std(ddof=1)
    z = (close - m) / s
    pos = pd.Series(0.0, index=close.index)
    state = 0.0
    for i in range(len(close)):
        zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = 0.0
            state = 0.0
            continue
        if state == 0.0:
            if zi <= -entry:
                state = 1.0
            elif zi >= entry:
                state = -1.0
        else:
            if abs(zi) <= exit:
                state = 0.0
        pos.iloc[i] = state
    return pos, z

def perf_table(name, equity, strat_rets, ann_factor, position):
    total_return = equity.iloc[-1] - 1.0 if len(equity) else np.nan
    md = max_drawdown(equity) if len(equity) else np.nan
    sh = sharpe(strat_rets, ann_factor)
    hr = hit_rate(strat_rets)
    to = turnover(position) / max(len(position.dropna()), 1)
    return pd.DataFrame([{
        "Strategy": name,
        "TotalReturn": total_return,
        "MaxDrawdown": md,
        "Sharpe": sh,
        "HitRate": hr,
        "TurnoverPerBar": to
    }])

def plot_microstructure(df):
    sp = spread_proxy(df)
    plt.figure(figsize=(10, 4))
    plt.plot(sp.index, sp.values)
    plt.title("Spread Proxy (High-Low)/Close")
    plt.xlabel("Time")
    plt.ylabel("Relative Spread Proxy")
    plt.grid(True)
    plt.show()
    return sp

def plot_intraday_returns(close):
    r = close.pct_change().dropna()
    plt.figure(figsize=(10, 4))
    plt.plot(r.index, r.values)
    plt.title("Intraday Returns (pct)")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.grid(True)
    plt.show()
    return r

def plot_vol_clustering(returns, idx):
    window = infer_bars_per_day(idx)
    vol = rolling_volatility(returns, window=window)
    plt.figure(figsize=(10, 4))
    plt.plot(vol.index, vol.values)
    plt.title(f"Rolling Volatility (window={window} bars)")
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.grid(True)
    plt.show()
    return vol

def plot_equity_curves(eq_ma, eq_mr, bh):
    plt.figure(figsize=(10, 5))
    plt.plot(bh.index, bh.values, label="Buy & Hold")
    plt.plot(eq_ma.index, eq_ma.values, label="MA Crossover (costed)")
    plt.plot(eq_mr.index, eq_mr.values, label="Mean Reversion (costed)")
    plt.title("Equity Curves")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_drawdowns(eq, title):
    peak = eq.cummax()
    dd = eq / peak - 1.0
    plt.figure(figsize=(10, 4))
    plt.plot(dd.index, dd.values)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.show()

def main():
    ticker = "SPY"
    period = "60d"
    interval = "5m"
    cost_per_turn = 0.0002
    slippage_per_turn = 0.0001

    df = download_intraday(ticker=ticker, period=period, interval=interval)
    close = df["Close"].copy()
    ann = bars_per_year(df.index)

    sp = plot_microstructure(df)
    r_pct = plot_intraday_returns(close)
    r_log = log_returns_from_close(close)
    vol = plot_vol_clustering(r_pct, df.index)

    pos_ma = ma_crossover_signals(close, fast=20, slow=50)
    strat_ma = apply_costs(pos_ma, r_pct, cost_per_turn=cost_per_turn, slippage_per_turn=slippage_per_turn)
    eq_ma, bh = equity_from_strategy(r_pct, strat_ma)

    pos_mr, z = mean_reversion_signals(close, lookback=infer_bars_per_day(df.index), entry=1.0, exit=0.2)
    strat_mr = apply_costs(pos_mr, r_pct, cost_per_turn=cost_per_turn, slippage_per_turn=slippage_per_turn)
    eq_mr, _ = equity_from_strategy(r_pct, strat_mr)

    plot_equity_curves(eq_ma, eq_mr, bh)
    plot_drawdowns(eq_ma, "Drawdown: MA Crossover (costed)")
    plot_drawdowns(eq_mr, "Drawdown: Mean Reversion (costed)")

    table_ma = perf_table("MA_Crossover_Costed", eq_ma, strat_ma, ann, pos_ma)
    table_mr = perf_table("MeanReversion_Costed", eq_mr, strat_mr, ann, pos_mr)
    table_bh = perf_table("BuyHold", bh, r_pct, ann, pd.Series(1.0, index=r_pct.index))

    summary = pd.concat([table_bh, table_ma, table_mr], ignore_index=True)

    out = pd.DataFrame({
        "Close": close,
        "ReturnPct": r_pct.reindex(close.index),
        "SpreadProxy": sp.reindex(close.index),
        "MA_Position": pos_ma.reindex(close.index),
        "MA_StrategyReturn": strat_ma.reindex(close.index),
        "MA_Equity": eq_ma.reindex(close.index),
        "MR_Position": pos_mr.reindex(close.index),
        "MR_StrategyReturn": strat_mr.reindex(close.index),
        "MR_Equity": eq_mr.reindex(close.index)
    })

    summary.to_csv("week13_summary.csv", index=False)
    out.to_csv("week13_timeseries.csv", index=True)

    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
