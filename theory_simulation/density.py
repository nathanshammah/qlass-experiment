import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm, ticker



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



cubename = str(input("filename: "))

rho = read_cube(f"{cubename}")
z_slice2 = rho[rho.shape[2] // 2, :, :]
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

fig, ax = plt.subplots(figsize=(10,6))
levels = 15

for i, symbol in enumerate(atom_symbols):
      ax.text(
          coords[i, 2],  # Slight offset in X so text doesn't overlap the dot
          coords[i, 1],  # Slight offset in Y

          symbol,
          fontsize=20,
          color="black",
          zorder=6,
          weight="bold",
      )
cont = ax.contourf(z_axis, y_axis, z_slice2, locator=ticker.LogLocator(), levels=levels, cmap=cm.coolwarm) #viridis, coolwarm

# add contour lines
cs = ax.contour(z_axis, y_axis, np.abs(np.log(z_slice2)), levels=levels, colors="white", linewidths=0.3)
cbar = plt.colorbar(cont)
cbar.ax.tick_params(labelsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
cbar.set_label("Electron density (a.u)", fontsize=20)
plt.tight_layout()
plt.savefig(f"{cubename.split('.')[0]}.svg", dpi=300, bbox_inches='tight')
plt.show()
