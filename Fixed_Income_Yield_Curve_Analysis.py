import numpy as np
import pandas as pd
from scipy.optimize import newton, curve_fit
import matplotlib.pyplot as plt

def bond_price(face, coupon_rate, ytm, maturity, freq=2):
    periods = maturity * freq
    coupon = face * coupon_rate / freq
    y = ytm / freq
    cash_flows = np.full(periods, coupon)
    cash_flows[-1] += face
    discount_factors = (1 + y) ** np.arange(1, periods + 1)
    return np.sum(cash_flows / discount_factors)

def yield_to_maturity(price, face, coupon_rate, maturity, freq=2):
    func = lambda y: bond_price(face, coupon_rate, y, maturity, freq) - price
    return newton(func, 0.05)

def macaulay_duration(face, coupon_rate, ytm, maturity, freq=2):
    periods = maturity * freq
    coupon = face * coupon_rate / freq
    y = ytm / freq
    times = np.arange(1, periods + 1)
    cash_flows = np.full(periods, coupon)
    cash_flows[-1] += face
    discount_factors = (1 + y) ** times
    pv = cash_flows / discount_factors
    return np.sum(times * pv) / np.sum(pv) / freq

def modified_duration(face, coupon_rate, ytm, maturity, freq=2):
    return macaulay_duration(face, coupon_rate, ytm, maturity, freq) / (1 + ytm / freq)

def convexity(face, coupon_rate, ytm, maturity, freq=2):
    periods = maturity * freq
    coupon = face * coupon_rate / freq
    y = ytm / freq
    times = np.arange(1, periods + 1)
    cash_flows = np.full(periods, coupon)
    cash_flows[-1] += face
    discount_factors = (1 + y) ** times
    pv = cash_flows / discount_factors
    return np.sum(pv * times * (times + 1)) / ((1 + y) ** 2 * np.sum(pv)) / (freq ** 2)

def dv01(face, coupon_rate, ytm, maturity):
    p1 = bond_price(face, coupon_rate, ytm + 0.0001, maturity)
    p2 = bond_price(face, coupon_rate, ytm - 0.0001, maturity)
    return (p2 - p1) / 2

def bootstrap_spot_rates(maturities, prices, face=100):
    spots = []
    for i, m in enumerate(maturities):
        if i == 0:
            r = (face / prices[i]) ** (1 / m) - 1
        else:
            discounted = sum(face * spots[j] / ((1 + spots[j]) ** maturities[j]) for j in range(i))
            r = ((face + discounted) / prices[i]) ** (1 / m) - 1
        spots.append(r)
    return np.array(spots)

def forward_rate(r1, r2, t1, t2):
    return ((1 + r2) ** t2 / (1 + r1) ** t1) ** (1 / (t2 - t1)) - 1

def nelson_siegel(t, beta0, beta1, beta2, tau):
    return beta0 + beta1 * ((1 - np.exp(-t / tau)) / (t / tau)) + beta2 * (((1 - np.exp(-t / tau)) / (t / tau)) - np.exp(-t / tau))

face = 100
coupon_rate = 0.05
ytm = 0.06
maturity = 5

price = bond_price(face, coupon_rate, ytm, maturity)
ytm_calc = yield_to_maturity(price, face, coupon_rate, maturity)
mac_dur = macaulay_duration(face, coupon_rate, ytm, maturity)
mod_dur = modified_duration(face, coupon_rate, ytm, maturity)
conv = convexity(face, coupon_rate, ytm, maturity)
dv = dv01(face, coupon_rate, ytm, maturity)

maturities = np.array([1, 2, 3, 4, 5])
prices = np.array([95, 92, 89, 87, 85])
spot_rates = bootstrap_spot_rates(maturities, prices)

forwards = [forward_rate(spot_rates[i], spot_rates[i + 1], maturities[i], maturities[i + 1]) for i in range(len(spot_rates) - 1)]

params, _ = curve_fit(nelson_siegel, maturities, spot_rates, maxfev=10000)
fitted_curve = nelson_siegel(maturities, *params)

results = pd.DataFrame({
    "Maturity": maturities,
    "Spot Rate": spot_rates,
    "Forward Rate": forwards + [np.nan],
    "Nelson-Siegel Fit": fitted_curve
})

plt.figure()
plt.plot(maturities, spot_rates, label="Spot Rates")
plt.plot(maturities, fitted_curve, label="Nelson-Siegel")
plt.legend()
plt.xlabel("Maturity")
plt.ylabel("Rate")
plt.show()

print("Bond Price:", price)
print("YTM:", ytm_calc)
print("Macaulay Duration:", mac_dur)
print("Modified Duration:", mod_dur)
print("Convexity:", conv)
print("DV01:", dv)
print(results)
