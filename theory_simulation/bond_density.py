import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker, colors


def read_cube(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Grid info
    nx = int(lines[3].split()[0])
    ny = int(lines[4].split()[0])
    nz = int(lines[5].split()[0])

    natoms = int(lines[2].split()[0])

    # data starts after header + atoms
    data_start = 6 + natoms

    data = []
    for line in lines[data_start:]:
        data.extend([float(x) for x in line.split()])

    rho = np.array(data).reshape((nx, ny, nz))
    return rho


from matplotlib.colors import LogNorm

cube_name = input("Input cube filename: ")

rho1 = read_cube("initial_pyscf.cube")

rho = read_cube(cube_name)
rho = rho - rho1
z_slice = rho[rho.shape[2] // 2, :, :]


bohr_to_ang = 0.529177

# 1. Cube metadata from your header
origin_y = -3.0  # Y origin
origin_z = -4.507057  # Z origin (from your header's 3rd column)
step_y = 0.030151
step_z = 0.045297  # Z step (from your header's 4th row)
ny, nz = 200, 200  # Shape of the Y and Z dimensions

# 2. Build Y and Z coordinate axes in Angstroms
y_axis = (origin_y + np.arange(ny) * step_y) * bohr_to_ang
z_axis = (origin_z + np.arange(nz) * step_z) * bohr_to_ang

# 3. Atom coordinates from your cube file (in Angstroms)
atom_coords_bohr = np.array(
    [[0.0, 0.0, -1.507057], [0.0, 0.0, 1.507057]]  # Li  # H
)
atom_symbols = ["Li", "H"]
coords = atom_coords_bohr * bohr_to_ang  # Shape: (2, 3)

#all_slices = [z_slice]
#vmin = min(np.min(arr) for arr in all_slices)
#vmax = max(np.max(arr) for arr in all_slices)
vmin = -0.12455000000000016
vmax = 0.023736300000000002

norm = colors.Normalize(vmin=vmin, vmax=vmax)

# --- PLOTTING ---
fig, ax = plt.subplots( figsize=(10, 6))
#levels = 40  # Back to a standard integer count, handled smoothly by LogNorm
levels = np.linspace(vmin, vmax, 60)
for i, symbol in enumerate(atom_symbols):
      ax.text(
          coords[i, 2],  # Slight offset in X so text doesn't overlap the dot
          coords[i, 0],  # Slight offset in Y

          symbol,
          fontsize=20,
          color="black",
          zorder=6,
          weight="bold",
      )
# Subplot 0
cont = ax.contourf(z_axis, y_axis, z_slice, levels=levels, norm=norm, cmap=cm.coolwarm)  #viridis, coolwarms
cs = ax.contour(z_axis, y_axis, z_slice, levels=levels, linewidths=0.1, colors="red")

ax.tick_params(axis='both', which='major', labelsize=20)

plt.tight_layout()

# --- COMMON COLORBAR CODE ---
fig.subplots_adjust(hspace=0.2, right=0.85)
cbar_ax = fig.add_axes([0.86, 0.16, 0.03, 0.8])
cbar = fig.colorbar(cont, cax=cbar_ax)

cbar.ax.tick_params(labelsize=20)


plt.savefig(f"{cube_name.split('.')[0]}_bond_density.svg", dpi=300, bbox_inches='tight')
# plt.savefig(f"/home/arajamani/QLASS_results/cubefiles/pp_mm_AO.svg", dpi=300, bbox_inches='tight')
plt.show()
