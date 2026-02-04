#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#Generalized Brière fit with pointwise CIs and an uncertainty band for the curve.
#Save the figure as EPS.
#For Eotetranychus kankitus
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================================================
# DATES
# ---------------------------------------------------------
# x: Temperatura (°C)
x = np.array([15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5])

# y: Observed rate (e.g., 1/days), eggs
#y = np.array([0.065, 0.094, 0.130, 0.18, 0.194,	0.236,	0.284,	0.29, 0.01])
# Development rate = 1 / larval duration in days
#y = np.array([0.122, 0.123, 0.25, 0.265, 0.397, 0.312, 0.424, 0.552, 0.])
# Development rate = 1 / protonymph duration in days
#y = np.array([0.156, 0.137, 0.303, 0.345, 0.402, 0.444, 0.505, 0.633, 0])
# Development rate = 1 / deutonymph duration in days
#y = np.array([0.117, 0.115, 0.252, 0.252, 0.427, 0.365, 0.448, 0.392, 0])
# Development rate = 1 / duration in days from egg to adult female
y = np.array([0.026, 0.029, 0.053, 0.062, 0.08,	 0.080,	0.099,	0.106])

# Pointwise uncertainty
sy = np.full_like(y, 0.01, dtype=float)   
# =========================================================
# 2) MODEL: Generalized Brière with free exponent p
#    r(T) = a * T * (T - Tmin) * (Tmax - T)^p, para Tmin < T < Tmax;
# ---------------------------------------------------------
def briere_generalized(T, a, Tmin, Tmax, p):
    T = np.asarray(T)
    r = np.zeros_like(T, dtype=float)
    mask = (T > Tmin) & (T < Tmax)
    # Avoid negative values due to numerical rounding:
    pos = Tmax - T[mask]
    pos[pos < 0] = 0.0
    r[mask] = a * T[mask] * (T[mask] - Tmin) * (pos ** p)
    # Soft clipping for numerical safety:
    r = np.where(np.isfinite(r), r, 0.0)
    r[r < 0] = 0.0
    return r

# =========================================================
# NONLINEAR FIT
# ---------------------------------------------------------
# Initial values and bounds
p0 = [4.0e-4, 10.9, 35.0, 0.4]          # a, Tmin, Tmax, p
bounds = ([1e-8,   10.8,  34.0, 0],  # Lower bounds
          [1e-2,  11,  36.0, 1])  # Upper bounds

# Weights: if you want to weight by uncertainty, use sigma=sy
# curve_fit will minimize the sum of (residual/sigma)^2
popt, pcov = curve_fit(
    briere_generalized, x, y,
    p0=p0, bounds=bounds, sigma=sy, absolute_sigma=True, maxfev=200000
)

perr = np.sqrt(np.diag(pcov))  # Standard error of each parameter

# =========================================================
# FITTED CURVE + 95% BAND (parametric samples)
# ---------------------------------------------------------
xx = np.linspace(0, 60, 800)
yy = briere_generalized(xx, *popt)

#Parameter sampling around popt using covariance pcov
#(If pcov is poorly conditioned, reduce n_samples or use jitter)
rng = np.random.default_rng(123)
n_samples = 1000

#Ensure positivity/basic coherence in samples (clip to bounds):
raw = rng.multivariate_normal(mean=popt, cov=pcov, size=n_samples)
raw[:, 0] = np.clip(raw[:, 0], bounds[0][0], bounds[1][0])  # a
raw[:, 1] = np.clip(raw[:, 1], bounds[0][1], bounds[1][1])  # Tmin
raw[:, 2] = np.clip(raw[:, 2], bounds[0][2], bounds[1][2])  # Tmax
raw[:, 3] = np.clip(raw[:, 3], bounds[0][3], bounds[1][3])  # p
# 
swap_mask = raw[:,2] <= raw[:,1]
raw[swap_mask, 2] = raw[swap_mask, 1] + 0.1

YY = np.array([briere_generalized(xx, *pars) for pars in raw])
yy_lo = np.percentile(YY, 2.5, axis=0)
yy_hi = np.percentile(YY, 97.5, axis=0)

# =========================================================
# PLOT (points with CIs and curve with band)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

# 95% error bars for the points (if sy = SD, 95% ≈ 1.96·SD) and confidence band
ci95 = 1.96 * sy
plt.errorbar(x, y, yerr=ci95, fmt='o', ms=5, capsize=3,
             label='Observed data (95% CI)')

# Fitted curve
plt.plot(xx, yy, lw=2, label='Fitted curve')
plt.xlabel("Temperature (°C)")
plt.ylabel("Development rate (1/days)")
plt.xlim(x.min()-2, x.max()+2)

ax = plt.gca()
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_color("black")
ax.set_facecolor("white")
plt.grid(True, color="lightgray", linewidth=0.6)
plt.legend()
plt.tight_layout()


plt.savefig("briere_fit_with_CI_adulto.eps", format="eps", dpi=600)
plt.xlim(0, 50)   
plt.show()
# ---------------------------------------------------------
names = ["a", "Tmin", "Tmax", "p"]
print("\nFitted parameters (generalized Brière):")
for n, v, e in zip(names, popt, perr):
    print(f"{n:5s} = {v:.6g} ± {e:.6g}")

# Basic metrics
y_hat = briere_generalized(x, *popt)
res = y - y_hat
ss_res = np.sum(res**2)
ss_tot = np.sum((y - y.mean())**2)
r2 = 1 - ss_res/ss_tot

# AIC/BIC assuming Gaussian errors with estimable variance
n, k = len(y), len(popt)
sigma2 = ss_res / n
AIC = n*np.log(sigma2) + 2*k
BIC = n*np.log(sigma2) + k*np.log(n)

print("\nFit metrics:")
print(f"R²     = {r2:.5f}")
print(f"AIC    = {AIC:.5f}")
print(f"BIC    = {BIC:.5f}")

# Numerical optimum within (Tmin, Tmax)
mask_opt = (xx > popt[1]) & (xx < popt[2])
T_opt = xx[mask_opt][np.argmax(yy[mask_opt])]
r_opt = yy[mask_opt].max()
print(f"Tóptimo (T_opt) ≈ {T_opt:.2f} °C, r(T_opt) = {r_opt:.4f}")
print(f"Prediction at 35 °C: r(35) = {briere_generalized(35, *popt):.4f}")
