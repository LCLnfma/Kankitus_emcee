#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 22:39:59 2026
Sensitivity analysis with PRCC (Tm and TM fixed); 
vary S_R and development/mortality parameters
@author: lauracruzadolima
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr, rankdata
from scipy.stats import t as student_t
import os

# ============================
# Physiological parameters (normal approximation)
# ============================
# Means and SDs from your table (Ullah fits)
PHYS = {
    "a_eL": (3.10e-4, 1.73e-5), "p_eL": (0.28, 0.04),
    "a_L1": (5.91e-4, 1.95e-5), "p_L1": (0.20, 0.03),
    "a_L2": (6.67e-4, 1.86e-5), "p_L2": (0.24, 0.02),
    "a_L3": (3.73e-4, 1.38e-5), "p_L3": (0.44, 0.03),
    "a_A" : (1.07e-4, 1.68e-5), "p_A" : (0.30, 0.12),

    # Mortality function parameters
    "k_m":    (0.93, 0.05),
    "Tmax_m": (23.26, 0.98),
    "rho_m":  (11.84, 2.21),
    "S_R": (0.76, 0.05),
}

# Fixed thresholds as decided (constants)
Tm_fixed = 10.8
TM_fixed = 35.0

y0 = [20, 0, 0, 0, 0, 10, 0]

# ============================
# Truncated sampling
# ============================
def sample_truncated_normal(mean, sd, size, low=None, high=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = rng.normal(mean, sd, size=size)
    if (low is None) and (high is None):
        return x
    # Rejection sampling simple
    for _ in range(50):
        bad = np.zeros_like(x, dtype=bool)
        if low is not None:
            bad |= (x < low)
        if high is not None:
            bad |= (x > high)
        if not bad.any():
            break
        x[bad] = rng.normal(mean, sd, size=bad.sum())
    # Final clipping for numerical stability
    if low is not None:
        x = np.maximum(x, low)
    if high is not None:
        x = np.minimum(x, high)
    return x

def draw_phys_params(N=2000, seed=123):
    rng = np.random.default_rng(seed)
    draws = {}
    for name, (mu, sd) in PHYS.items():
        # Reasonable bounds
        if name.startswith("a_"):
            draws[name] = sample_truncated_normal(mu, sd, N, low=1e-10, high=None, rng=rng)
        elif name.startswith("p_"):
            draws[name] = sample_truncated_normal(mu, sd, N, low=0.01, high=5.0, rng=rng)
        elif name == "k_m":
            draws[name] = sample_truncated_normal(mu, sd, N, low=0.01, high=2.0, rng=rng)
        elif name == "Tmax_m":
            draws[name] = sample_truncated_normal(mu, sd, N, low=0.0, high=50.0, rng=rng)
        elif name == "rho_m":
            draws[name] = sample_truncated_normal(mu, sd, N, low=0.1, high=50.0, rng=rng)
        elif name == "S_R":
            draws[name] = sample_truncated_normal(mu, sd, N, low=0.001, high=1.0, rng=rng) 
        else:
            draws[name] = rng.normal(mu, sd, N)

    X = pd.DataFrame(draws)
    return X

# ============================
# PRCC (Spearman partial)
# ============================
def prcc_spearman(X: pd.DataFrame, y: np.ndarray):
    Xv = X.to_numpy(dtype=float)
    n, p = Xv.shape

    # rank-transform (Spearman)
    Xr = np.apply_along_axis(rankdata, 0, Xv)
    yr = rankdata(y)

    prcc = np.zeros(p)
    pval = np.zeros(p)

    # Regression matrix
    for j in range(p):
        idx = [k for k in range(p) if k != j]
        Z = Xr[:, idx]
        Z = np.column_stack([np.ones(n), Z])

        # residual of X_j on Z
        beta_x, *_ = np.linalg.lstsq(Z, Xr[:, j], rcond=None)
        rx = Xr[:, j] - Z @ beta_x

        # residual of and on Z
        beta_y, *_ = np.linalg.lstsq(Z, yr, rcond=None)
        ry = yr - Z @ beta_y

        r = np.corrcoef(rx, ry)[0, 1]
        dfree = n - (p) - 1  # n - (#predictores Z) - 1
        tstat = r * np.sqrt(dfree / (1 - r**2 + 1e-12))
        pv = 2 * (1 - student_t.cdf(np.abs(tstat), dfree))
        prcc[j] = r
        pval[j] = pv

    out = pd.DataFrame({
        "Parameter": X.columns,
        "PRCC": prcc,
        "p_value": pval,
        "abs_PRCC": np.abs(prcc),
    }).sort_values("abs_PRCC", ascending=False).drop(columns=["abs_PRCC"])
    return out

# ============================
# Function to run the Rossini model and extract peaks
# ============================
# Auxiliary functions
# =========================
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
    """
    Thermal fecundity curve:
    beta(T) = a * exp(1 + (b - T)/k - exp((b - T)/k))
    """
    z = (b - T) / k
    return a * np.exp(1.0 + z - np.exp(z))

# Rossini dynamics
def Rossini(t, y, a_beta, b_beta, k_beta, c2, a2, G_list, T_data, M_arr, S_R):
    i = int(min(len(T_data)-1, max(0, round(t))))
    T = T_data[i]
    g_EL, g_L1, g_L2, g_L3, g_A = G_list[0][i], G_list[1][i], G_list[2][i], G_list[3][i], G_list[4][i]
    m =  M_arr[i]
    beta_T = beta_fun(T, a_beta, b_beta, k_beta)
    X_e, X_L1, X_L2, X_L3, X_Am, X_Af1, X_Af2 = y
    natalidad = beta_T * (X_Af1+c2*X_Af2)
    return [
        natalidad - g_EL * X_e - m * X_e,
        g_EL * X_e - g_L1 * X_L1 - m * X_L1,
        g_L1 * X_L1 - g_L2 * X_L2 - m * X_L2,
        g_L2 * X_L2 - g_L3 * X_L3 - m * X_L3,
        (1 - S_R) * g_L3 * X_L3 - g_A * X_Am - m * X_Am,
        S_R * g_L3 * X_L3 - X_Af1 + a2 * X_Af2,
        (1 - g_A) * X_Af1 - m * X_Af1 - g_A * X_Af2 - m * X_Af2 - a2 * X_Af2
    ]

def build_G_list_from_draw(draw_row, T_data):
    G_list = [
        G(T_data, draw_row["a_eL"], TM_fixed, Tm_fixed, draw_row["p_eL"]),
        G(T_data, draw_row["a_L1"], TM_fixed, Tm_fixed, draw_row["p_L1"]),
        G(T_data, draw_row["a_L2"], TM_fixed, Tm_fixed, draw_row["p_L2"]),
        G(T_data, draw_row["a_L3"], TM_fixed, Tm_fixed, draw_row["p_L3"]),
        G(T_data, draw_row["a_A"],  TM_fixed, Tm_fixed, draw_row["p_A"]),
    ]
    return G_list

def build_M_arr_from_draw(draw_row, T_data):
    par = (draw_row["k_m"], draw_row["Tmax_m"], draw_row["rho_m"])
    return mortfun(par, T_data)

def simulate_and_peaks(X_phys, fixed_field_params, y0, dias_totales, T_data):
    N = len(X_phys)
    peak_mag = np.full(N, np.nan)
    peak_time = np.full(N, np.nan)

    for i in range(N):
        row = X_phys.iloc[i]

        G_list = build_G_list_from_draw(row, T_data)
        M_arr = build_M_arr_from_draw(row, T_data)

        sol = solve_ivp(
            Rossini, [0, dias_totales[-1]], y0, t_eval=dias_totales,
            args=(fixed_field_params["a_beta"], fixed_field_params["b_beta"], fixed_field_params["k_beta"],
                  fixed_field_params["c2"], fixed_field_params["a2"],
                  G_list, T_data, M_arr, row["S_R"]),
            method="LSODA", rtol=1e-6, atol=1e-8
        )

        if not sol.success or sol.y is None:
            continue

        A = sol.y[1] + sol.y[2] + sol.y[3] + sol.y[4] + sol.y[5] + sol.y[6]
        A = np.maximum(A, 0.0)
        Amax = np.nanmax(A)
        idx = np.where(np.isclose(A, Amax))[0]
        imax = int(np.round(idx.mean()))
        peak_mag[i] = float(A[imax])
        peak_time[i] = float(dias_totales[imax])

    ok = np.isfinite(peak_mag) & np.isfinite(peak_time)
    return X_phys.loc[ok].reset_index(drop=True), peak_mag[ok], peak_time[ok]
# %%
#For the plot
def prcc_barplot2(df_prcc, ax=None, title="", ylabel="PRCC",
                  order="abs", show_values=True, ylim=(-1, 1),
                  color="forestgreen", labels_map=None,
                  rotation=45, tick_fontsize=12):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))

    d = df_prcc.copy()
    if order == "abs":
        d["abs"] = np.abs(d["PRCC"])
        d = d.sort_values("abs", ascending=False).drop(columns="abs")

    x = np.arange(len(d))
    y = d["PRCC"].to_numpy()

    ax.axhline(0, color="black", linewidth=1)
    ax.bar(x, y, color=color)

    ax.set_xticks(x)

    if labels_map is None:
        xt = d["Parameter"].tolist()
    else:
        xt = [labels_map.get(p, p) for p in d["Parameter"].tolist()]

    ax.set_xticklabels(xt, rotation=rotation, ha="right", fontsize=tick_fontsize)

    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=20)
    ax.set_ylim(*ylim)

    if show_values:
        for i, v in enumerate(y):
            ax.text(i, v + (0.03 if v >= 0 else -0.05),
                    f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=16)

    ax.grid(axis="y", alpha=0.25)
    return ax

# %%
# ============================
# ============================
#Fix field parameters using the global posterior mean:
    
base_dir = "/Users/lauracruzadolima/Documents/POSDOC/Registros de conteos/fit_Parameters/Rossini_t_urticae"

df_A = pd.read_csv(os.path.join(base_dir, "E_kankitus.csv"), parse_dates=["Fechas"])
df_T = pd.read_csv(os.path.join(base_dir, "temperatura_tepe.csv"), parse_dates=["Fecha"])

df_A["date"] = pd.to_datetime(df_A["Fechas"]).dt.normalize()
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
dias_obs = df.index[df["Promedio"].notna()].to_numpy()
n_obs = len(A_obs)
print("n_obs =", n_obs)   
    
post_global = pd.read_csv("posterior_Global_with_p.csv")
fixed = post_global[["a_beta","b_beta","k_beta","c2","a2","p_nb"]].mean().to_dict()

X_phys = draw_phys_params(N=2000, seed=123)
X_ok, peak_mag, peak_time = simulate_and_peaks(X_phys, fixed, y0, dias_totales, T_data)
print("N inicial:", len(X_phys), "N ok:", len(X_ok))

# PRCC for peak magnitude and timing:
prcc_mag  = prcc_spearman(X_ok, peak_mag)
prcc_time = prcc_spearman(X_ok, peak_time)

print(prcc_mag.head(10))
print(prcc_time.head(10))

prcc_mag.to_csv("PRCC_peak_magnitude.csv", index=False)
prcc_time.to_csv("PRCC_peak_timing.csv", index=False)

# %%
LABELS = {
    "k_m": r"$k$",
    "Tmax_m": r"$T_{\max}$",
    "rho_m": r"$\rho_T$",

    "a_eL": r"$a^{eL}$", "p_eL": r"$p^{eL}$",
    "a_L1": r"$a^{L_1}$", "p_L1": r"$p^{L_1}$",
    "a_L2": r"$a^{L_2}$", "p_L2": r"$p^{L_2}$",
    "a_L3": r"$a^{L_3}$", "p_L3": r"$p^{L_3}$",
    "a_A":  r"$a^{A}$",    "p_A":  r"$p^{A}$",
    "S_R": r"$S_R$",
}

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharey=True)

prcc_barplot2(prcc_mag,  ax=axes[0], title="a) Peak magnitude",
              ylabel="PRCC", color="forestgreen",
              labels_map=LABELS, rotation=0, tick_fontsize=20)

prcc_barplot2(prcc_time, ax=axes[1], title="b) Peak timing",
              ylabel="PRCC", color="forestgreen",
              labels_map=LABELS, rotation=0, tick_fontsize=20)

for ax in axes:
    ax.set_xlabel("Physiological parameters", fontsize=20)

plt.tight_layout()
plt.ylim(-1.1, 1.1)
plt.subplots_adjust(bottom=0.01)  # << clave para que NO se amontonen
plt.savefig("PRCC_bars.pdf", format="eps", bbox_inches="tight")
plt.show()