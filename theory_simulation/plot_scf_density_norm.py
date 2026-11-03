import matplotlib.pyplot as plt
import re
from sympy.codegen.ast import integer
import numpy as np
from pathlib import Path
import pandas as pd


def lsv_file(totit, l_df):
    nrm_array = []

    for t in range(1, totit+1):
        l_sv = pd.read_csv(f"SV_AO_Density_SCF_iteration_{t}_VQE.csv", index_col=0)
        sub = l_sv.to_numpy() - l_df.to_numpy()
        nrm = np.linalg.norm(sub)
        nrm_array.append(nrm)

    return nrm_array

def lsv_file_dft(totit, l_df):
    nrm_array = []

    for t in range(1, totit+1):
        l_sv = pd.read_csv(f"DFT_AO_Density_SCF_iteration_{t}_VQE.csv", index_col=0)
        sub = l_sv.to_numpy() - l_df.to_numpy()
        nrm = np.linalg.norm(sub)
        nrm_array.append(nrm)

    return nrm_array

def den_trail(totit, l_df, trail=10):
    den_t = pd.DataFrame()
    for trails in range(1, trail+1):
        nrm_array = []
        for t in range(1, totit+1):
            l_sv = pd.read_csv(f"snoise/AO_Density_Trail_{trails}_SCF_iteration_{t}_VQE.csv", index_col=0)
            sub = l_sv.to_numpy() - l_df.to_numpy()
            nrm = np.linalg.norm(sub)
            nrm_array.append(nrm)
        nrm_df = pd.DataFrame({f'{trails}': nrm_array})
        den_t = pd.concat([den_t, nrm_df], axis=1)
    return den_t



trail = 10

df = {}
df_mean = pd.DataFrame()
for i in range(1, trail+1):
    fn = pd.read_csv(f"snoise/TotalEnergy_Trail_{i}" + ".csv", index_col=0)
    fn_e = fn["Total_energy"][:6]
    df[i] = fn_e
    df_mean = pd.concat([df_mean,fn_e], axis=1)

df_file_LiH = "DFT_AO_Density_SCF_iteration_7_VQE.csv" # From pyscf DFT manual routine
l_df_LiH = pd.read_csv(df_file_LiH, index_col=0)



total_iteration = 6

snoise_dennorm = den_trail(total_iteration, l_df_LiH, trail)


tot_norm = lsv_file(total_iteration, l_df_LiH)
tot_norm_DFT = lsv_file_dft(6, l_df_LiH)
sv_d = pd.read_csv("LiH_SCF_SV_total_energies.csv")
liH = float(7.79215308635209) # From pyscf
sv_d['dE'] = sv_d['Energy'].diff().fillna(0.00000000001)
pyscf_energy = np.array([-7.77352639397485, -7.76939222170814, -7.79205331109308, -7.79215302941812,
                -7.79215308492036, -7.7921530863489]) # From pyscf


fig, ax = plt.subplots(2,1, figsize=(10,10), sharex=True)
x = np.arange(1, 7)
ax[0].plot(x, np.abs(sv_d['Energy'] + liH), lw=4., color='r', marker="o", ms=12, label="Statevector")
ax[0].plot(x, np.abs(pyscf_energy + liH), ls=":",lw=4., color='g', marker="o", ms=12, label="PySCF")
ax[0].plot(x, np.abs(df_mean.mean(axis=1) + liH), lw=4., ls="--",color='purple', marker="o", ms=12, label="mean \n(shot noise)")
for i in range(1, trail+1):
    ax[0].plot(x, np.abs(df[i].values + liH), ls='-', color='purple',alpha=0.2)


ax[0].set_yscale('log')
ax[0].tick_params(axis='both', which='major', labelsize=20)
ax[0].set_ylabel(r"$\Delta E$ (Ha)", fontsize=20)



ax[1].plot(x, tot_norm, lw=4., color='r', marker="o", ms=12, label="Statevector")
ax[1].plot(x, tot_norm_DFT, ls=":", lw=4., color='g', marker="o", ms=12, label="PySCF")
ax[1].plot(x, snoise_dennorm.mean(axis=1), ls="--", lw=4, color='purple',marker="o", ms=12, label="mean \n(shot noise)")
for i in range(1, trail+1):
    ax[1].plot(x, np.abs(snoise_dennorm[f"{i}"].values), ls='-', color='purple',alpha=0.2)

ax[1].tick_params(axis='both', which='major', labelsize=20)
ax[1].set_xlabel("SCF iteration", fontsize=20)
ax[1].set_ylabel(r"$||\Delta \rho||$ (a.u)", fontsize=20)
ax[1].set_yscale("log")
ax[1].set_ylim(0.000001, 10)
ax[0].set_ylim(0.00000001, 10)

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], lw=4, color='r',marker="o", ms=12, label="Statevector"),
                Line2D([0], [0], lw=4, color='g', ls=":",marker="o", ms=12, label="Pyscf"),
                Line2D([0], [0], color='purple',ls="-",label=r"Shot noise", alpha=0.2),
                Line2D([0], [0], lw=4, color='purple',ls="--",label="Mean \n(shot noise)"),
                ]
ax[1].legend(handles=custom_lines, loc='lower left', fontsize=20)
plt.tight_layout()
# fig.savefig("LiH_SCF_Density.svg", dpi=300)
plt.show()
