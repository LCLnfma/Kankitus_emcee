# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 13:05:34 2025
survival for mortality
For Eotetranychus kankitus
@author: lauracruzadolima
"""
#%%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import odr
from scipy.stats.distributions import chi2
import os
import sys


#%% Read the data
os.chdir("/Users/lauracruzadolima/Documents/POSDOC/Registros de conteos/fit_Parameters/Rossini_t_urticae")
Data = pd.read_csv("data_kankitus.txt", sep="\t", header=None)
Data.columns = ["x", "y", "err_x", "err_y"]

x = Data['x'].values
y = Data['y'].values
err_x = Data['err_x'].values
err_y = Data['err_y'].values

#%% Define the mortality function (modified Gompertz type)
def mortfun(par, x):
    a, b, k = par
    try:
        z = (b - x) / k
        result = 1 - a * np.exp(1 + z - np.exp(z))
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            return np.ones_like(x) * 1e6  # Penalize if there are invalid values
        return result
    except Exception:
        return np.ones_like(x) * 1e6  

#%% Configure and run ODR
mort_model = odr.Model(mortfun)
data = odr.RealData(x, y, sx=err_x, sy=err_y)
odr_fit = odr.ODR(data, mort_model, beta0=[0.5, 20, 5])  # Initial values

out = odr_fit.run()

# Validate fit results
if not np.all(np.isfinite(out.beta)):
    print("Error! The fit did not converge")
    sys.exit()

best_fit = out.beta
errors = out.sd_beta

#%% Confidence band
num_sigma = 2  # by default: 95% confidence

x_fun = np.linspace(0, 40, 1000)
fit_curve = mortfun(best_fit, x_fun)

params_upper = best_fit + num_sigma * errors
params_lower = best_fit - num_sigma * errors

fit_upper = mortfun(params_upper, x_fun)
fit_lower = mortfun(params_lower, x_fun)

#%% Fit statistics
fitted_vals = mortfun(best_fit, x)
residuals = y - fitted_vals
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
ndf = len(x) - len(best_fit)

chi_sq = np.sum((residuals**2) / (err_y**2))
p_value = 1 - chi2.cdf(chi_sq, ndf)

# Information criteria
AIC = len(x) * np.log(ss_res / len(x)) + 2 * len(best_fit)
BIC = len(x) * np.log(ss_res / len(x)) + len(best_fit) * np.log(len(x))

#%% Results
print("\nFit result (Gompertz-type function):\n")
param_names = ["a", "b", "k"]
for i in range(len(best_fit)):
    print(f"{param_names[i]} = {best_fit[i]:.6f} ± {errors[i]:.6f}")
print(f"R² = {r_squared:.5f}")
print(f"Chi² = {chi_sq:.5f}")
print(f"P-valor = {p_value:.5e}")
print(f"NDF = {ndf}")
print(f"AIC = {AIC:.5f}")
print(f"BIC = {BIC:.5f}\n")

#%% Plot
fig = go.Figure()

# Original data with error bars
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Observed data',
                         error_y=dict(type='data', array=err_y, visible=True)))

# Fitted curve
fig.add_trace(go.Scatter(x=x_fun, y=fit_curve, mode='lines', name='Fitted curve'))

fig.update_layout(
    title="Mortality function adjustment",
    xaxis_title="Temperature (°C)",
    yaxis_title="Rate Mortality",
    legend=dict(x=0.01, y=0.99),
    plot_bgcolor="white",   # Plot background
    paper_bgcolor="white"   # Outer background
)

# Gray grid and black borders on the plot frame
fig.update_xaxes(showgrid=True, gridcolor="lightgray",
                 linecolor="black", mirror=True)
fig.update_yaxes(showgrid=True, gridcolor="lightgray",
                 linecolor="black", mirror=True)

fig.show()

fig.write_html("fit_mortality.html", auto_open=True)