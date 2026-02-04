#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 21:59:31 2025

@author: lauracruzadolima
"""

"""
# Rossini adapted for E. kankitus
# Priors: normal for a_beta, b_beta, k_beta, c2, and G1<-2(a2)
# Uniform prior for p
# Likelihood: Negative Binomial (NB)
# MCMC: emcee (6 parameters)
# Corner plot
# 95% bands: 2.5–97.5 percentiles of simulated trajectories
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import solve_ivp
from scipy.special import gammaln
from scipy.optimize import differential_evolution
import corner
import os
import time
import multiprocessing
import arviz as az
import emcee
##########################################
plt.rcParams.update({
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12
})
########################################3


# =========================
# Configuración general
# =========================
A_BETA_BOUNDS = (1.0, 3.5)   # Amplitude of beta(T)
B_BETA_BOUNDS = (25.0, 35.0)   # Optimal temperature
K_BETA_BOUNDS = (6.0, 15.0)   # Curve width
C2_BOUNDS     = (0.1, 2.0)    # Relative fecundity Af2
A2_BOUNDS     = (0.01, 3.0)   # Range for G1<-2
P_BOUNDS = (0.02, 0.40)   # uniforme (elige rango plausible)

USE_PARALLEL = True
N_PROCS = min(8, multiprocessing.cpu_count())
LIKELIHOOD = "nb"

EPS = 1e-8

# --- Normal priors  ---
A_BETA_MEAN, A_BETA_SD = 2.9, 0.5
B_BETA_MEAN, B_BETA_SD = 27.0, 2.0
K_BETA_MEAN, K_BETA_SD = 10.0, 2.0
C2_MEAN,     C2_SD     = 1.0, 0.4
A2_MEAN,     A2_SD     = 1.5, 0.5


# =========================
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

# =========================
# Likelihoods
# =========================
def loglik_nb_mu_p(y_obs, mu_obs, p_nb):
    """
    Negative Binomial with mean = mu_obs, var = mu_obs / p_nb.
    We use the NumPy parametrization:
      Y ~ NB(r_i, p_nb)
      E[Y]   = r_i * (1-p_nb) / p_nb = mu_i
      Var[Y] = r_i * (1-p_nb) / p_nb**2 = mu_i / p_nb
    where r_i = p_nb * mu_i / (1 - p_nb).
    """
    p = p_nb
    # Avoid numerical issues
    mu_obs = np.maximum(mu_obs, EPS)
    p = min(max(p, 1e-6), 1-1e-6)

    r = p * mu_obs / (1.0 - p)     # r_i por tiempo
    r = np.maximum(r, EPS)

    # log p(Y=y | r, p) = lgamma(y+r) - lgamma(r) - lgamma(y+1)
    #                    + r log(p) + y log(1-p)
    return np.sum(
        gammaln(y_obs + r) - gammaln(r) - gammaln(y_obs + 1.0)
        + r * np.log(p) + y_obs * np.log(1.0 - p)
    )

def log_likelihood(params, y0, dias_totales, dias_obs, A_obs,
                   G_list, T_data, M_arr, S_R):
    a_beta, b_beta, k_beta, c2, a2, p_nb = params

    sol = solve_ivp(
        Rossini, [0, dias_totales[-1]], y0, t_eval=dias_totales,
        args=(a_beta, b_beta, k_beta, c2, a2, G_list, T_data, M_arr, S_R),
        method="LSODA", rtol=1e-6, atol=1e-8
    )
    if not sol.success:
        return -np.inf

    A_mod = sol.y[1] + sol.y[2] + sol.y[3] + sol.y[4] + sol.y[5] + sol.y[6]
    if np.any((dias_obs < 0) | (dias_obs >= len(A_mod))):
        return -np.inf

    mu_obs = np.maximum(A_mod[dias_obs], EPS)
    y_obs  = np.maximum(A_obs,       EPS)

    # NB with mean = mu_obs y p = NB_P (fixed)
    return loglik_nb_mu_p(y_obs, mu_obs, p_nb)


# =========================
# Priors
# =========================

def _normal_logpdf(x, mean, sd):
    z = (x - mean) / sd
    return -0.5 * (z*z) - np.log(sd * np.sqrt(2*np.pi))


def log_prior(theta):
    a_beta, b_beta, k_beta, c2, a2, p = theta

    # bounds duros
    if not (A_BETA_BOUNDS[0] <= a_beta <= A_BETA_BOUNDS[1]): return -np.inf
    if not (B_BETA_BOUNDS[0] <= b_beta <= B_BETA_BOUNDS[1]): return -np.inf
    if not (K_BETA_BOUNDS[0] <= k_beta <= K_BETA_BOUNDS[1]): return -np.inf
    if not (C2_BOUNDS[0] <= c2     <= C2_BOUNDS[1]):         return -np.inf
    if not (A2_BOUNDS[0] <= a2     <= A2_BOUNDS[1]):         return -np.inf
    # p uniform  [P_BOUNDS]
    if not (P_BOUNDS[0] <= p <= P_BOUNDS[1]): return -np.inf

    lp  = _normal_logpdf(a_beta, A_BETA_MEAN, A_BETA_SD)
    lp += _normal_logpdf(b_beta, B_BETA_MEAN, B_BETA_SD)
    lp += _normal_logpdf(k_beta, K_BETA_MEAN, K_BETA_SD)
    lp += _normal_logpdf(c2,     C2_MEAN,     C2_SD)
    lp += _normal_logpdf(a2,     A2_MEAN,     A2_SD)
    return lp

def log_posterior(theta, y0, dias_totales, dias_obs, A_obs, G_list, T_data, M_arr, S_R):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, y0, dias_totales, dias_obs, A_obs, G_list, T_data, M_arr, S_R)
    return lp + ll if np.isfinite(ll) else -np.inf

# --- emcee top-level (picklable) ---
def log_post_emcee(theta, y0, dias_totales, dias_obs, A_obs, G_list, T_data, M_arr, S_R):
    return log_posterior(theta, y0, dias_totales, dias_obs, A_obs, G_list, T_data, M_arr, S_R)

def _reflect(x, lo, hi):
    if lo == hi: return lo
    span = hi - lo
    y = lo + np.abs((x - lo) % (2 * span))
    if y > hi: y = hi - (y - hi)
    return y
# =========================
# Main
# =========================
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    np.random.seed(42)
    inicio = time.time()

    S_R = 0.76

    # -- ---
    base_dir = "/Users/lauracruzadolima/Documents/POSDOC/Registros de conteos/fit_Parameters/Rossini_t_urticae"
    df_A = pd.read_csv(os.path.join(base_dir, "T_urticae_var4.csv"), parse_dates=["Fechas"]) #Change according to the cultivar: use E_kankitus_var1 or omit var1 for the global model
    df_T = pd.read_csv(os.path.join(base_dir, "temperatura_tepe.csv"), parse_dates=["Fecha"])

    df_A["date"] = pd.to_datetime(df_A["Fechas"]).dt.normalize()
    df_T["date"] = pd.to_datetime(df_T["Fecha"]).dt.normalize()
    df_A = df_A.sort_values("date").drop_duplicates(subset="date", keep="last")
    df_T = df_T.sort_values("date").drop_duplicates(subset="date", keep="last")

    df = pd.merge(df_T[["date", "Temperatura"]], df_A[["date", "Promedio"]], on="date", how="left")
    df["dias"] = (df["date"] - df["date"].min()).dt.days

    T_data       = df["Temperatura"].to_numpy()
    dias_totales = df["dias"].to_numpy()
    A_obs        = df.loc[df["Promedio"].notna(), "Promedio"].to_numpy()
    dias_obs     = df.index[df["Promedio"].notna()].to_numpy()
    n_obs = len(A_obs)
    print("n_obs =", n_obs)
    
    fechas_totales = df["date"].to_numpy()
    fechas_obs     = df.loc[df["Promedio"].notna(), "date"].to_numpy()
    

    print(f"T_data: {T_data.shape}, días totales: {dias_totales.shape}")
    print(f"N observaciones: {A_obs.shape[0]} en días {dias_obs[:10]}...")

    # --- Calibrated thermal rates ---
    G_list = [
        G(T_data, 0.000309651, 35, 10.8, 0.283549),
        G(T_data, 0.000591476, 35, 10.8, 0.202752),
        G(T_data, 0.000666829, 35, 10.8, 0.235524),
        G(T_data, 0.000373163, 35, 10.8, 0.436374),
        G(T_data, 0.000108623, 35, 10.8, 0.300295)
    ]
    par_mort = (0.931427, 23.264781, 11.843984)
    M_arr = mortfun(par_mort, T_data)

    # --- Initial conditions ---
    y0 = [20, 0, 0, 0, 0, 10, 0]
    a_beta_opt = 2.25   
    b_beta_opt = 28.26  
    k_beta_opt = 9.88  
    c2_opt     = 0.67  
    a2_opt     = 1.41   
    p_init = 0.05
    # ---------------------------
    # 2) MCMC with emcee
    # ---------------------------
    ndim = 6
    theta_map = [a_beta_opt, b_beta_opt, k_beta_opt, c2_opt, a2_opt, p_init]
    scale = np.array([0.08 * max(1e-3, abs(theta_map[0])),
                      0.08 * max(1e-3, abs(theta_map[1])),
                      0.08 * max(1e-3, abs(theta_map[2])),
                      0.08 * max(1e-3, abs(theta_map[3])),
                      0.08 * max(1e-3, abs(theta_map[4])),
                      0.03
    ])

    nwalkers = 4 * ndim
    nsteps = 5000
    burn_in = 1000
    p0 = np.array([theta_map + np.random.normal(0, scale) for _ in range(nwalkers)])
    for w in range(nwalkers):
        p0[w][0] = _reflect(p0[w][0], *A_BETA_BOUNDS)
        p0[w][1] = _reflect(p0[w][1], *B_BETA_BOUNDS)
        p0[w][2] = _reflect(p0[w][2], *K_BETA_BOUNDS)
        p0[w][3] = _reflect(p0[w][3], *C2_BOUNDS)
        p0[w][4] = _reflect(p0[w][4], *A2_BOUNDS)
        p0[w][5] = _reflect(p0[w][5], *P_BOUNDS)


    log_args = (y0, dias_totales, dias_obs, A_obs, G_list, T_data, M_arr, S_R)
    if USE_PARALLEL:
        with multiprocessing.Pool(processes=N_PROCS) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_emcee, pool=pool, args=log_args)
            state = sampler.run_mcmc(p0, burn_in, progress=True)  # burn-in
            sampler.reset()
            sampler.run_mcmc(state, nsteps, progress=True)       
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_emcee, args=log_args)
        state = sampler.run_mcmc(p0, burn_in, progress=True)
        sampler.reset()
        sampler.run_mcmc(state, nsteps, progress=True)

    print("Mean acceptance rate (emcee):", np.mean(sampler.acceptance_fraction))
# %% 
    chains = sampler.get_chain()     # (nwalkers, nsteps, 6)
    arr = chains
    flat = chains.reshape(-1, ndim)
    df_samples = pd.DataFrame(flat, columns=["a_beta", "b_beta", "k_beta", "c2", "a2", "p_nb"])
    df_samples.to_csv("posterior_F70_with_p_0.csv", index=False)
    a_beta_draws = np.transpose(chains[:, :, 0], (1, 0))
    b_beta_draws = np.transpose(chains[:, :, 1], (1, 0))
    k_beta_draws = np.transpose(chains[:, :, 2], (1, 0))
    c2_draws     = np.transpose(chains[:, :, 3], (1, 0))
    a2_draws     = np.transpose(chains[:, :, 4], (1, 0))
    p_draws      = np.transpose(chains[:, :, 5], (1, 0))
    
    # --- Posterior summary ---
    mean_params = df_samples.mean()
    stderr = df_samples.std()
    print("\nEstimated parameters (posterior):")
    for name, mean, std in zip(df_samples.columns, mean_params, stderr):
        print(f"{name} = {mean:.6f} ± {std:.6f}")
    a_beta_mean, b_beta_mean, k_beta_mean, c2_mean, a2_mean, p_mean = mean_params.values

    # --- Simulación con medias ---
    sol_m = solve_ivp(
        Rossini, [0, dias_totales[-1]], y0,
        args=(a_beta_mean, b_beta_mean, k_beta_mean, c2_mean, a2_mean, G_list, T_data, M_arr, S_R),
        method="BDF", rtol=2e-6, atol=1e-8, max_step=1.0, dense_output=True
    )
    if sol_m.success and (sol_m.sol is not None):
        Yall = sol_m.sol(dias_totales)
        A_mod = Yall[1] + Yall[2] + Yall[3] + Yall[4] + Yall[5] + Yall[6]
    else:
        sol2 = solve_ivp(Rossini, [0, dias_totales[-1]], y0, t_eval=dias_totales,
                         args=(a_beta_mean, b_beta_mean, k_beta_mean, c2_mean, a2_mean, G_list, T_data, M_arr, S_R),
                         method="LSODA", rtol=2e-6, atol=1e-8)
        A_mod = sol2.y[1] + sol2.y[2] + sol2.y[3] + sol2.y[4] + sol2.y[5] + sol2.y[6]

    # Métricas (loglik en el mismo modo seleccionado)
    A_mod_obs = A_mod[dias_obs]
    resid = A_obs - A_mod_obs
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((A_obs - np.mean(A_obs))**2)
    r_squared = 1 - (ss_res / ss_tot)
    ll = log_likelihood([a_beta_mean, b_beta_mean, k_beta_mean, c2_mean, a2_mean, p_mean], y0, dias_totales, dias_obs, A_obs, G_list, T_data, M_arr, S_R)
    k = 6
    AIC = 2 * k - 2 * ll
    print(f"\nR²: {r_squared:.6f}   |   AIC ({LIKELIHOOD}): {AIC:.6f}")
    # Ver los datos que entran en el cálculo de R²
    #r2_df = pd.DataFrame({
    #    "date": fechas_obs,
    #    "day_index": dias_obs,
    #    "observed": A_obs,
    #    "model": A_mod_obs
    #    })
    #print("\nDatos usados para el cálculo de R² (observado vs modelo en días de muestreo):")
    #print(r2_df)

    # %%
# Corner 
    flat_for_corner = df_samples.sample(n=min(4000, len(df_samples)), random_state=123)

    fig = corner.corner(
        flat_for_corner.values,
        labels=[r"$a_{\beta}$", r"$b_{\beta}$", r"$k_{\beta}$",
                r"$c_{2}$", r"$G_{1\leftarrow 2}$", r"$p$"],
        quantiles=[0.16, 0.5, 0.84],   # Vertical lines
        show_titles=False,            # <- NO numbers on top
        label_kwargs={"fontsize": 24},
        )

    fig.suptitle("Posterior distributions of model parameters", fontsize=18, y=1.02)
    fig.tight_layout()
    plt.show()
    fig.savefig("corner_posterior_F70.eps", format="eps", bbox_inches="tight")
    plt.close(fig)
    
# %%
    # Observado vs Modelo
    plt.figure()
    plt.plot(dias_totales, A_mod, label="Model")
    plt.plot(dias_obs, A_obs, 'o', label="Field data")
    plt.xlabel("Time (days)")
    plt.ylabel(r"The average number of $\it{E.\ kankitus}$ per leaf")
    plt.title(f"Simulated vs observed  |  Likelihood={LIKELIHOOD}")
    plt.legend(); plt.grid(); plt.show()
# %%

    # Diagnostics ArviZ
    idata = az.from_dict(
        posterior={
            "a_beta": a_beta_draws,
            "b_beta": b_beta_draws,
            "k_beta": k_beta_draws,
            "c2": c2_draws,
            "a2": a2_draws,
            "p_nb": p_draws
            }
        )
    print("\nR-hat:")
    print("\nR-hat:\n", az.rhat(idata))
    print("\nESS:\n", az.ess(idata))
    #az.plot_trace(idata, var_names=["a_beta", "b_beta", "k_beta", "c2", "a2", "p_nb"]); plt.tight_layout(); plt.show()
    #az.plot_autocorr(idata, var_names=["a_beta", "b_beta", "k_beta", "c2", "a2", "p_nb"]); plt.tight_layout(); plt.show()
# %%
    # ============================
    #  95% uncertainty band
    # ============================
    print("\nGenerando banda de incertidumbre (percentiles 2.5 y 97.5)...")
    n_simulaciones = 300
    simulaciones = []
    for _ in range(n_simulaciones):
        a_beta_i, b_beta_i, k_beta_i, c2_i, a2_i, p_i = df_samples.sample(n=1)[
            ["a_beta","b_beta","k_beta","c2","a2","p_nb"]].values[0]
        try:
            sol_i = solve_ivp(
                Rossini, [0, dias_totales[-1]], y0, t_eval=dias_totales,
                args=(a_beta_i, b_beta_i, k_beta_i, c2_i, a2_i, G_list, T_data, M_arr, S_R),
                method="LSODA", rtol=3e-6, atol=1e-8
            )
            if sol_i.success:
                A_i = sol_i.y[1] + sol_i.y[2] + sol_i.y[3] + sol_i.y[4] + sol_i.y[5] + sol_i.y[6]
                simulaciones.append(A_i)
        except Exception:
            continue
    simulaciones = np.array(simulaciones)
    media_sim = np.mean(simulaciones, axis=0)
    lower_band = np.percentile(simulaciones, 2.5, axis=0)
    upper_band = np.percentile(simulaciones, 97.5, axis=0)
    # %%
# ============================================================
#Figure: uncertainty band + point trajectory with DATES
# ============================================================

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # --- 95% credible band ---
    ax.fill_between(
        fechas_totales,
        lower_band,
        upper_band,
        alpha=0.2,
        label="95% credible interval"
        )

# --- Posterior predictive mean ---
    ax.plot(
        fechas_totales,
        media_sim,
        linewidth=4,
        label="Posterior predictive mean"
        )

# --- Observed data ---
    ax.scatter(
        fechas_obs,
        A_obs,
        s=60,
        label="Observed data"
        )

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    plt.xticks(rotation=0)

    ax.tick_params(axis='both', labelsize=18)
    ax.set_ylim(0, 200)
    plt.xlabel("Days", fontsize=18)
    plt.ylabel(r"Mites per leaf (all mobile stages)", fontsize=18)
    plt.title("Population dynamics (F70)", fontsize=20)

    plt.grid(alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig("F70.pdf", dpi=300)
    plt.show()
    
