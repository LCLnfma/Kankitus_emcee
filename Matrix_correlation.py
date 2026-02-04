#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:48:50 2026
Correlation matrix
@author: lauracruzadolima
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

base_dir = "/Users/lauracruzadolima/Documents/POSDOC/Registros de conteos/fit_Parameters/Rossini_t_urticae"

df_A = pd.read_csv(os.path.join(base_dir, "E_kankitus.csv"), parse_dates=["Fechas"])
df_T = pd.read_csv(os.path.join(base_dir, "temperatura_tepe.csv"), parse_dates=["Fecha"])

# Parameters in the desired order
PARAMS = ["a_beta", "b_beta", "k_beta", "c2", "a2", "p_nb"]

# Load global and cultivar-specific posteriors
post_global = pd.read_csv("posterior_F70_with_p.csv")


# Spearman correlation
C = post_global[PARAMS].corr(method="spearman")

LATEX_LABELS = {
    "a_beta": r"$a_\beta$",
    "b_beta": r"$b_\beta$",
    "k_beta": r"$k_\beta$",
    "c2":     r"$c_2$",
    "a2":   r"$G_{1\leftarrow 2}$",
    "p_nb":      r"$p$"
}

def plot_corr_heatmap(C, title, outpath):
    labels = [LATEX_LABELS[c] for c in C.columns]
    A = C.to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(A, vmin=-1, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)

    ax.set_title(title, fontsize=13)

    # Numerical values
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i,j]:.2f}",
                    ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.show()
    
plot_corr_heatmap(
    C,
    title="F70",
    outpath="corr_F70.pdf"
)
