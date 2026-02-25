# spr_heatmap_kretschmann_sio2.py
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Paths (SPPPy/ is in parent directory)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from SPPPy import ExperimentSPR, Layer, MaterialDispersion, nm  # noqa: E402

BASE_CSV = PROJECT_ROOT / "PermittivitiesBase.csv"

# ------------------------------------------------------------
# User parameters (easy to tweak)
# ------------------------------------------------------------
ANGLE_DEG = 45.0  # fixed incidence angle (you can adjust)
POLARIZATION = "p"  # SPR => p-polarization
AG_THICKNESS_NM = 55.0  # Ag thickness
SIO2_THICKNESS_NM = np.arange(0, 31, 1)  # 0..30 nm, step 1 nm
WAVELENGTH_NM = np.arange(400, 901, 1)  # 400..900 nm, step 1 nm

# Colormap:
# For R in [0,1], a perceptually uniform sequential map is best for linear gradation.
CMAP_NAME = "viridis"

# ------------------------------------------------------------
# Materials from your database
# ------------------------------------------------------------
bk7 = MaterialDispersion("BK7", base_file=str(BASE_CSV))
ag = MaterialDispersion("Ag", base_file=str(BASE_CSV))
sio2 = MaterialDispersion("SiO2", base_file=str(BASE_CSV))
air_n = 1.0  # Air as constant refractive index

# ------------------------------------------------------------
# Build Kretschmann structure: BK7 / Ag / SiO2 / Air
# ------------------------------------------------------------
exp = ExperimentSPR(polarization=POLARIZATION)

prism_layer = Layer(bk7, 0, "BK7 prism")  # semi-infinite incident medium
ag_layer = Layer(ag, AG_THICKNESS_NM * nm, "Ag")  # 55 nm silver
sio2_layer = Layer(sio2, 0 * nm, "SiO2")  # variable thickness
air_layer = Layer(air_n, 0, "Air")  # semi-infinite output medium

exp.add(prism_layer)
exp.add(ag_layer)
exp.add(sio2_layer)
exp.add(air_layer)

# ------------------------------------------------------------
# Compute R(lambda, d_SiO2) map
# ------------------------------------------------------------
wl_m = WAVELENGTH_NM * nm
R_map = np.zeros((len(SIO2_THICKNESS_NM), len(WAVELENGTH_NM)), dtype=float)

for i, d_nm in enumerate(SIO2_THICKNESS_NM):
    sio2_layer.thickness = d_nm * nm  # cache-aware update inside SPPPy
    R_spectrum = np.asarray(exp.R(wl_range=wl_m, angle=ANGLE_DEG), dtype=float)
    R_map[i, :] = R_spectrum
    print(
        f"[{i+1:02d}/{len(SIO2_THICKNESS_NM)}] SiO2 = {d_nm:>2.0f} nm  "
        f"Rmin={R_spectrum.min():.4f}  Rmax={R_spectrum.max():.4f}"
    )

# Optional sanity clipping (numerical noise guard)
R_map = np.clip(R_map, 0.0, 1.0)

# ------------------------------------------------------------
# Plot 2D scalar field (pcolormesh)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

mesh = ax.pcolormesh(
    WAVELENGTH_NM,
    SIO2_THICKNESS_NM,
    R_map,
    shading="auto",
    cmap=CMAP_NAME,
    vmin=0.0,
    vmax=1.0,
)

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Reflectance R")

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("SiO$_2$ thickness (nm)")
ax.set_title(
    f"Kretschmann SPR map: BK7 / Ag({AG_THICKNESS_NM:.0f} nm) / SiO$_2$ / Air, "
    f"θ = {ANGLE_DEG:.1f}°, pol = {POLARIZATION}"
)

# Keep exact axes ranges for reproducibility
ax.set_xlim(WAVELENGTH_NM.min(), WAVELENGTH_NM.max())
ax.set_ylim(SIO2_THICKNESS_NM.min(), SIO2_THICKNESS_NM.max())

plt.show()
