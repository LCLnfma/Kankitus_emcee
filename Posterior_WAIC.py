#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 08:19:22 2025
Reads saved chain files to avoid rerunning all programs.
Calculates WAIC, LOO, confidence intervals, and posterior plots for the five cultivars.
@author: lauracruzadolima
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.special import gammaln
from scipy.integrate import solve_ivp
import os
import datetime as dt
import corner
import arviz as az
#============================
EPS = 1e-12
#NB_P = 0.2 
# ============================
# Cargar datos de temperatura y tiempos
# ============================
base_dir = "/Users/lauracruzadolima/Documents/POSDOC/Registros de conteos/fit_Parameters/Rossini_t_urticae"

df_A = pd.read_csv(os.path.join(base_dir, "E_kankitus.csv"), parse_dates=["Fechas"])
df_T = pd.read_csv(os.path.join(base_dir, "temperatura_tepe.csv"), parse_dates=["Fecha"])

df_A["date"]= pd.to_datetime(df_A["Fechas"]).dt.normalize()
df_T["date"] = pd.to_datetime(df_T["Fecha"]).dt.normalize()
df_A = df_A.sort_values("date").drop_duplicates(subset="date", keep="last")
df_T = df_T.sort_values("date").drop_duplicates(subset="date", keep="last")

df = pd.merge(df_T[["date", "Temperatura"]],
              df_A[["date", "Promedio"]],
              on="date", how="left")
df["dias"] = (df["date"] - df["date"].min()).dt.days

T_data       = df["Temperatura"].to_numpy()
dias_totales = df["dias"].to_numpy()
A_obs    = df.loc[df["Promedio"].notna(), "Promedio"].to_numpy()
dias_obs = df.index[df["Promedio"].notna()].to_numpy()  # índices que coinciden con dias_totales
n_obs = len(A_obs)
print("n_obs =", n_obs)

#month labels in the same positions as total_days
# etiquetas de meses en las mismas posiciones que dias_totales
month_labels = [
    "Jan", "Jan", "Jan", "Feb", "Mar", "Mar", "Apr", "Apr", "May", "May",
    "Jun", "Jun", "Jul", "Jul", "Aug", "Aug", "Sep", "Sep", "Oct"
]
# %%
# ============================
# Rossini model functions
# ============================
def G(T, a, b, c, p):
    T = np.asarray(T, dtype=float)
    base = np.clip(b - T, 0.0, None)
    r = a * T * (T - c) * (base ** p)
    r[~np.isfinite(r)] = 0.0
    r[r < 0] = 0.0
    return r

def mortfun(par, x):
    a, b, k = par
    z = (b - x) / k
    z = np.clip(z, -10, 10)
    return 1 - a * np.exp(1 + z - np.exp(z))

def beta_fun(T, a, b, k):
    z = (b - T) / k
    return a * np.exp(1.0 + z - np.exp(z))

def Rossini(t, y, a_beta, b_beta, k_beta, c2, a2, G_list, T_data, M_arr, S_R):
    i = int(min(len(T_data)-1, max(0, round(t))))
    T = T_data[i]
    g_EL, g_L1, g_L2, g_L3, g_A = (
        G_list[0][i], G_list[1][i], G_list[2][i], G_list[3][i], G_list[4][i]
    )
    m = M_arr[i]
    beta_T = beta_fun(T, a_beta, b_beta, k_beta)
    X_e, X_L1, X_L2, X_L3, X_Am, X_Af1, X_Af2 = y
    natalidad = beta_T * (X_Af1 + c2 * X_Af2)
    return [
        natalidad - g_EL * X_e - m * X_e,
        g_EL * X_e - g_L1 * X_L1 - m * X_L1,
        g_L1 * X_L1 - g_L2 * X_L2 - m * X_L2,
        g_L2 * X_L2 - g_L3 * X_L3 - m * X_L3,
        (1 - S_R) * g_L3 * X_L3 - g_A * X_Am - m * X_Am,
        S_R * g_L3 * X_L3 - X_Af1 + a2 * X_Af2,
        (1 - g_A) * X_Af1 - m * X_Af1 - g_A * X_Af2 - m * X_Af2 - a2 * X_Af2
    ]
#==================================================
# Likelihood
def loglik_nb_pointwise(y, mu, p_nb):
    y  = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), EPS)

    p = float(p_nb)
    p = min(max(p, 1e-9), 1 - 1e-9)

    r = p * mu / (1 - p)
    r = np.maximum(r, EPS)

    ll = (
        gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
        + r * np.log(p) + y * np.log(1.0 - p)
    )
    return ll

def pointwise_loglik_theta(theta, y0, dias_totales, dias_obs, A_obs,
                           G_list, T_data, M_arr, S_R):
    a_beta, b_beta, k_beta, c2, a2, p_nb = theta

    sol = solve_ivp(
        Rossini, [0, dias_totales[-1]], y0, t_eval=dias_totales,
        args=(a_beta, b_beta, k_beta, c2, a2, G_list, T_data, M_arr, S_R),
        method="LSODA", rtol=1e-6, atol=1e-8
    )
    if (not sol.success) or (sol.y is None):
        return -np.inf * np.ones(len(A_obs), dtype=float)

    A_mod = sol.y[1] + sol.y[2] + sol.y[3] + sol.y[4] + sol.y[5] + sol.y[6]
    mu_obs = np.maximum(A_mod[dias_obs], EPS)

    return loglik_nb_pointwise(A_obs, mu_obs, p_nb)

#==================================================
#==================================================
#WAIC
def waic_from_posterior_df(df_samples, y0, dias_totales, dias_obs, A_obs,
                          G_list, T_data, M_arr, S_R,
                          n_draws=2000, seed=123):
    """
    Compute WAIC using ArviZ from a CSV file of posterior samples.
    """
    rng = np.random.default_rng(seed)
    n_total = len(df_samples)
    n_draws = min(n_draws, n_total)

    draw_idx = rng.choice(n_total, size=n_draws, replace=False)
    thetas = df_samples.iloc[draw_idx][["a_beta","b_beta","k_beta","c2","a2","p_nb"]].to_numpy()

    n_obs = len(A_obs)
    loglik = np.empty((1, n_draws, n_obs), dtype=float)  # (chains=1, draws, obs)

    for j in range(n_draws):
        ll_i = pointwise_loglik_theta(
            thetas[j], y0, dias_totales, dias_obs, A_obs,
            G_list, T_data, M_arr, S_R
        )
        loglik[0, j, :] = ll_i

    idata = az.from_dict(log_likelihood={"y": loglik})

    waic_res = az.waic(idata, pointwise=True)
    return waic_res
#==================================================

# Temperature-dependent development rates (the same as those used in the fitting)
G_list = [
    G(T_data, 0.000309651, 35, 10.8, 0.283549),
    G(T_data, 0.000591476, 35, 10.8, 0.202752),
    G(T_data, 0.000666829, 35, 10.8, 0.235524),
    G(T_data, 0.000373163, 35, 10.8, 0.436374),
    G(T_data, 0.000108623, 35, 10.8, 0.300295)
]
par_mort = (0.931427, 23.264781, 11.843984)
M_arr = mortfun(par_mort, T_data)
S_R = 0.76
y0 = [20, 0, 0, 0, 0, 10, 0]

# ============================
# Function: posterior predictive mean
# ============================
def posterior_predictive_mean(df_samples, n_draws=300):
    simulaciones = []
    cols = ["a_beta","b_beta","k_beta","c2","a2"] 
    
    for _ in range(n_draws):
        a_beta_i, b_beta_i, k_beta_i, c2_i, a2_i = df_samples[cols].sample(n=1).values[0]
        sol = solve_ivp(
            Rossini,
            [0, dias_totales[-1]],
            y0,
            t_eval=dias_totales,
            args=(a_beta_i, b_beta_i, k_beta_i, c2_i, a2_i, G_list, T_data, M_arr, S_R),
            method="LSODA",
            rtol=3e-6,
            atol=1e-8
        )
        if sol.success:
            A_i = sol.y[1] + sol.y[2] + sol.y[3] + sol.y[4] + sol.y[5] + sol.y[6]
            simulaciones.append(A_i)

    simulaciones = np.array(simulaciones)
    return simulaciones.mean(axis=0)
# ==================================
# 2.5% and 97.5% credible quantiles
# ==================================
def summarize_posterior(csv_path, label):
    df = pd.read_csv(csv_path)
    params = ["a_beta","b_beta","k_beta","c2","a2", "p_nb"]

    out = []
    for p in params:
        x = df[p].to_numpy()
        mean = np.mean(x)
        sd   = np.std(x, ddof=1)
        q025, q50, q975 = np.quantile(x, [0.025, 0.5, 0.975])
        out.append([label, p, mean, sd, q50, q025, q975])
    return pd.DataFrame(out, columns=["Fit","Param","Mean","SD","Median","q2.5","q97.5"])

# ============================
#Load posteriors and compute curves
# ============================

post_global = pd.read_csv("posterior_Global_with_p.csv")
post_OA     = pd.read_csv("posterior_OroAzteca_with_p.csv")
post_DM     = pd.read_csv("posterior_DM_with_p.csv")
post_AR     = pd.read_csv("posterior_Arkansas_with_p.csv")
post_F70    = pd.read_csv("posterior_F70_with_p.csv")

A_global = posterior_predictive_mean(post_global)
A_OA     = posterior_predictive_mean(post_OA)
A_DM     = posterior_predictive_mean(post_DM)
A_AR     = posterior_predictive_mean(post_AR)
A_F70    = posterior_predictive_mean(post_F70)


# ============================
# ============================
fecha_inicio = dt.datetime(2024, 4, 1)   

fechas = [fecha_inicio + dt.timedelta(days=int(d)) for d in dias_totales]
# %%
# ============================
# Comparative plot
# ============================
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(fechas, A_global, label="Global", lw=4)
ax.plot(fechas, A_OA,     label="Oro Azteca", lw=2)
ax.plot(fechas, A_DM,     label="Diamante Mejorado", lw=2)
ax.plot(fechas, A_AR,     label="Arkansas", lw=2)
ax.plot(fechas, A_F70,    label="F70", lw=2)

# --- X-axis formatted as 01-Apr, 01-May, etc. 
ax.xaxis.set_major_locator(mdates.MonthLocator())       
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))  # 01-Apr

plt.xticks(rotation=0)

ax.tick_params(axis='both', labelsize=18)
ax.set_ylim(0, 180)

ax.set_xlabel("Days", fontsize=18)
ax.set_ylabel(r"Mites per leaf (all mobile stages)", fontsize=18)
ax.set_title(r"Population dynamics of $\it{E. kankitus}$", fontsize=20)

ax.grid(alpha=0.3)
ax.legend(fontsize=17)

plt.tight_layout()
plt.savefig("population_dynamics_varieties_with_p.pdf", dpi=300)
plt.show()
# %% WAIC
posts = {
    "Global": post_global,
    "Oro Azteca": post_OA,
    "Diamante Mejorado": post_DM,
    "Arkansas": post_AR,
    "F70": post_F70
}

for name, post in posts.items():
    res = waic_from_posterior_df(
        post, y0, dias_totales, dias_obs, A_obs,
        G_list, T_data, M_arr, S_R,
        n_draws=2000, seed=123
    )
    print(f"{name:18s} | elpd_waic={float(res.elpd_waic):8.2f} | p_waic={float(res.p_waic):6.2f} | WAIC={-2*float(res.elpd_waic):8.2f}")
# %%LOO
#LOO
def loo_from_posterior_df(df_samples, y0, dias_totales, dias_obs, A_obs,
                          G_list, T_data, M_arr, S_R,
                          n_draws=2000, seed=123):

    rng = np.random.default_rng(seed)
    n_total = len(df_samples)
    n_draws = min(n_draws, n_total)

    draw_idx = rng.choice(n_total, size=n_draws, replace=False)
    thetas = df_samples.iloc[draw_idx][["a_beta","b_beta","k_beta","c2","a2","p_nb"]].to_numpy()

    n_obs = len(A_obs)
    loglik = np.empty((1, n_draws, n_obs), dtype=float)  # (chains=1, draws, obs)

    for j in range(n_draws):
        ll_i = pointwise_loglik_theta(
            thetas[j], y0, dias_totales, dias_obs, A_obs,
            G_list, T_data, M_arr, S_R
        )
        loglik[0, j, :] = ll_i
    idata = az.from_dict(
    posterior={"dummy": np.zeros((1, n_draws))},
    log_likelihood={"y": loglik}
    )
    loo_res = az.loo(idata, pointwise=True)
    return loo_res

posts = {
    "Global": post_global,
    "Oro Azteca": post_OA,
    "Diamante Mejorado": post_DM,
    "Arkansas": post_AR,
    "F70": post_F70
}

for name, post in posts.items():
    res = loo_from_posterior_df(
        post, y0, dias_totales, dias_obs, A_obs,
        G_list, T_data, M_arr, S_R,
        n_draws=2000, seed=123
    )
    print(f"{name:18s} | elpd_loo={float(res.elpd_loo):8.2f} | p_loo={float(res.p_loo):6.2f} | LOOIC={-2.0*float(res.elpd_loo):8.2f}")
    pk = np.asarray(res.pareto_k)
    print(f"{name:18s} | max_k={pk.max():.2f} | k>0.7={(pk>0.7).sum()} | k>1={(pk>1).sum()}")
    pk = np.asarray(res.pareto_k)
    idx = np.where(pk > 0.7)[0]
    print(idx, pk[idx])
    print("dias_obs influyentes:", dias_obs[idx])
    print("A_obs influyentes:", A_obs[idx])
    fechas_obs = df.loc[df["Promedio"].notna(), "date"].to_numpy()
    print("fechas influyentes:", fechas_obs[idx])

# %%
# ============================
# Summaries: mean, SD, and 95% CrI (2.5% and 97.5%)
# ============================
summ_global = summarize_posterior("posterior_Global_with_p.csv", "Global")
summ_OA     = summarize_posterior("posterior_OroAzteca_with_p.csv", "Oro Azteca")
summ_DM     = summarize_posterior("posterior_DM_with_p.csv", "Diamante Mejorado")
summ_AR     = summarize_posterior("posterior_Arkansas_with_p.csv", "Arkansas")
summ_F70    = summarize_posterior("posterior_F70_with_p.csv", "F70")

summary_all = pd.concat([summ_global, summ_OA, summ_DM, summ_AR, summ_F70], ignore_index=True)

print(summary_all)

summary_all.to_csv("posterior_summary_95CrI_with_p.csv", index=False)

summary_all["Mean_95CrI"] = summary_all.apply(
    lambda r: f"{r['Mean']:.3f} [{r['q2.5']:.3f}, {r['q97.5']:.3f}]",
    axis=1
)

print(summary_all[["Fit","Param","Mean_95CrI"]])
summary_all[["Fit","Param","Mean_95CrI"]].to_csv(
    "posterior_summary_mean_95CrI_compact.csv",
    index=False
)

