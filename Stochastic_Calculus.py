import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt, log, exp

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def black_scholes_call_put(S0, K, r, sigma, T):
    if T <= 0 or sigma <= 0 or S0 <= 0 or K <= 0:
        raise ValueError("Require S0>0, K>0, sigma>0, T>0")
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call = S0 * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    put = K * exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)
    return call, put

def brownian_paths(T=1.0, n_steps=252, n_paths=10, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    dW = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    W = np.cumsum(dW, axis=1)
    W = np.hstack([np.zeros((n_paths, 1)), W])
    t = np.linspace(0.0, T, n_steps + 1)
    return t, W, dt

def gbm_exact_paths(S0=100.0, mu=0.10, sigma=0.20, T=1.0, n_steps=252, n_paths=10, seed=42):
    t, W, dt = brownian_paths(T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    S = S0 * np.exp((mu - 0.5 * sigma * sigma) * t + sigma * W)
    return t, S, dt

def euler_maruyama_gbm(S0=100.0, mu=0.10, sigma=0.20, T=1.0, n_steps=252, n_paths=10, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    dW = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for k in range(n_steps):
        S[:, k + 1] = S[:, k] + mu * S[:, k] * dt + sigma * S[:, k] * dW[:, k]
    t = np.linspace(0.0, T, n_steps + 1)
    return t, S, dt

def ito_demo_lnS(S_paths, mu, sigma, dt):
    S = S_paths
    dS = S[:, 1:] - S[:, :-1]
    S_prev = S[:, :-1]
    dW_hat = (dS / S_prev - mu * dt) / sigma
    dlnS = np.log(S[:, 1:] / S[:, :-1])
    drift_theory = (mu - 0.5 * sigma * sigma) * dt
    diff_theory_var = (sigma * sigma) * dt
    dlnS_centered = dlnS - drift_theory
    mean_dlnS = np.mean(dlnS)
    var_dlnS = np.var(dlnS, ddof=1)
    mean_dW_hat = np.mean(dW_hat)
    var_dW_hat = np.var(dW_hat, ddof=1)
    return {
        "mean_dlnS": mean_dlnS,
        "var_dlnS": var_dlnS,
        "drift_theory_per_step": drift_theory,
        "diff_theory_var_per_step": diff_theory_var,
        "mean_dW_hat": mean_dW_hat,
        "var_dW_hat": var_dW_hat,
        "dlnS_centered": dlnS_centered.ravel()
    }

def monte_carlo_call_gbm(S0, K, r, sigma, T, n_paths=200000, n_steps=252, seed=123):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.normal(0.0, 1.0, size=(n_paths, n_steps))
    increments = (r - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z
    logS = np.log(S0) + np.cumsum(increments, axis=1)
    ST = np.exp(logS[:, -1])
    payoff = np.maximum(ST - K, 0.0)
    disc = np.exp(-r * T)
    price = disc * np.mean(payoff)
    se = disc * np.std(payoff, ddof=1) / np.sqrt(n_paths)
    ci = (price - 1.96 * se, price + 1.96 * se)
    return price, se, ci, ST

def main():
    S0 = 100.0
    mu = 0.10
    r = 0.05
    sigma = 0.20
    T = 1.0
    n_steps = 252
    n_paths_plot = 20

    tW, W, dtW = brownian_paths(T=T, n_steps=n_steps, n_paths=n_paths_plot, seed=7)
    plt.figure(figsize=(10, 5))
    for i in range(W.shape[0]):
        plt.plot(tW, W[i], alpha=0.7)
    plt.title("Brownian Motion Paths")
    plt.xlabel("t")
    plt.ylabel("W(t)")
    plt.grid(True)
    plt.show()

    tE, S_exact, dt = gbm_exact_paths(S0=S0, mu=mu, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths_plot, seed=8)
    tM, S_em, _ = euler_maruyama_gbm(S0=S0, mu=mu, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths_plot, seed=8)

    plt.figure(figsize=(10, 5))
    for i in range(S_exact.shape[0]):
        plt.plot(tE, S_exact[i], alpha=0.6)
    plt.title("GBM Exact Simulation Paths")
    plt.xlabel("t")
    plt.ylabel("S(t)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    for i in range(S_em.shape[0]):
        plt.plot(tM, S_em[i], alpha=0.6)
    plt.title("GBM Euler–Maruyama Simulation Paths")
    plt.xlabel("t")
    plt.ylabel("S(t)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(tE, np.mean(S_exact, axis=0), label="Exact mean")
    plt.plot(tM, np.mean(S_em, axis=0), label="Euler–Maruyama mean")
    plt.title("Exact vs Euler–Maruyama: Mean Path")
    plt.xlabel("t")
    plt.ylabel("E[S(t)] (sample)")
    plt.grid(True)
    plt.legend()
    plt.show()

    demo = ito_demo_lnS(S_exact, mu=mu, sigma=sigma, dt=dt)
    plt.figure(figsize=(10, 5))
    plt.hist(demo["dlnS_centered"], bins=60, density=True)
    plt.title("Ito Demo: d(ln S) centered by theoretical drift")
    plt.xlabel("d(ln S) - (mu - 0.5*sigma^2) dt")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

    K = 100.0
    bs_call, bs_put = black_scholes_call_put(S0=S0, K=K, r=r, sigma=sigma, T=T)
    mc_call, mc_se, mc_ci, ST = monte_carlo_call_gbm(S0=S0, K=K, r=r, sigma=sigma, T=T, n_paths=150000, n_steps=n_steps, seed=11)

    plt.figure(figsize=(10, 5))
    plt.hist(ST, bins=80, density=True)
    plt.title("Distribution of Terminal Price S(T) from GBM Monte Carlo")
    plt.xlabel("S(T)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

    print("=== Stochastic Calculus Mini-Project Summary ===")
    print(f"Parameters: S0={S0}, mu={mu}, r={r}, sigma={sigma}, T={T}, steps={n_steps}")
    print("")
    print("Ito Lemma Check using ln(S):")
    print(f"Empirical mean dlnS per step: {demo['mean_dlnS']:.8f}")
    print(f"Theoretical drift per step:   {demo['drift_theory_per_step']:.8f}")
    print(f"Empirical var dlnS per step:  {demo['var_dlnS']:.8f}")
    print(f"Theoretical var per step:     {demo['diff_theory_var_per_step']:.8f}")
    print(f"Empirical mean dW_hat:        {demo['mean_dW_hat']:.8f}")
    print(f"Empirical var dW_hat:         {demo['var_dW_hat']:.8f}")
    print("")
    print("Black–Scholes vs Monte Carlo (European Call):")
    print(f"Black–Scholes Call: {bs_call:.6f}")
    print(f"Black–Scholes Put:  {bs_put:.6f}")
    print(f"Monte Carlo Call:   {mc_call:.6f}")
    print(f"MC Std Error:       {mc_se:.6f}")
    print(f"MC 95% CI:          [{mc_ci[0]:.6f}, {mc_ci[1]:.6f}]")
    print("")
    print("Interpretation (short):")
    print("1) Brownian motion paths show variance growing with time.")
    print("2) GBM exact vs Euler–Maruyama should be close at small dt, but EM has discretization error.")
    print("3) Ito demo confirms d ln S has drift (mu - 0.5 sigma^2) dt and variance sigma^2 dt.")
    print("4) Monte Carlo option price should align with Black–Scholes within the CI.")

if __name__ == "__main__":
    main()
