import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Fig. 15 replication (Barchiesi & Otto, 2013)
# TMM / Fresnel for a single finite layer (p-polarization)
# -----------------------------

# Physical parameters from the article
LAMBDA0_NM = 546.1  # nm
EPS1 = 3.6168 + 0j  # prism
EPS_AIR = 1.0
EPS_AG = -10.9204 + 0.8334j  # silver at 546.1 nm

# Thicknesses from Fig. 15 legend/caption
OTTO_THICKNESSES_NM = [424.8, 531.0, 637.2]
KRETSCH_THICKNESSES_NM = [38.4, 48.0, 57.6]

# Angle axis chosen to match Fig. 15 scale for overlay checks
THETA_DEG = np.linspace(29.0, 40.0, 4001)


def physical_sqrt(z: np.ndarray) -> np.ndarray:
    """
    Square root with a branch choice suitable for optics:
    enforce Im(sqrt) >= 0; if Im ~ 0, enforce Re >= 0.
    Works elementwise on complex numpy arrays.
    """
    w = np.sqrt(z.astype(np.complex128))
    mask_flip = (np.imag(w) < -1e-14) | (
        (np.abs(np.imag(w)) <= 1e-14) & (np.real(w) < 0)
    )
    w[mask_flip] = -w[mask_flip]
    return w


def fresnel_r_p(eps_i, eps_j, w_i, w_j):
    """
    p-polarized Fresnel reflection coefficient at interface i|j
    using dimensionless z-components w_i, w_j.
    """
    return (eps_j * w_i - eps_i * w_j) / (eps_j * w_i + eps_i * w_j)


def reflectance_three_medium_p(theta_deg, eps1, eps2, eps3, d2_nm, lambda0_nm):
    """
    Reflectance R=|r|^2 for a 3-medium stack:
        medium 1 (incident, semi-infinite)
        medium 2 (finite thickness d2)
        medium 3 (substrate, semi-infinite)
    p-polarization.
    """
    theta = np.deg2rad(theta_deg)
    u = np.sqrt(eps1) * np.sin(theta)  # dimensionless kx/k0

    w1 = physical_sqrt(eps1 - u**2)
    w2 = physical_sqrt(eps2 - u**2)
    w3 = physical_sqrt(eps3 - u**2)

    r12 = fresnel_r_p(eps1, eps2, w1, w2)
    r23 = fresnel_r_p(eps2, eps3, w2, w3)

    k0 = 2.0 * np.pi / lambda0_nm  # nm^-1
    phase = np.exp(2j * k0 * w2 * d2_nm)

    r = (r12 + r23 * phase) / (1.0 + r12 * r23 * phase)
    R = np.abs(r) ** 2
    return R.real


def save_curve_csv(path, theta_deg, R):
    """
    Save exactly two columns as requested: R and angle (deg).
    Column order: R, angle_deg
    """
    data = np.column_stack([R, theta_deg])
    np.savetxt(path, data, delimiter=",", header="R,angle_deg", comments="")


def main():
    out_dir = "fig15_csv"
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.0), constrained_layout=True)

    # ---------- Otto configuration ----------
    # medium 2 = air (finite gap), medium 3 = Ag (semi-infinite)
    ax = axes[0]
    for d_nm in OTTO_THICKNESSES_NM:
        R = reflectance_three_medium_p(
            THETA_DEG,
            eps1=EPS1,
            eps2=EPS_AIR,
            eps3=EPS_AG,
            d2_nm=d_nm,
            lambda0_nm=LAMBDA0_NM,
        )
        ax.plot(THETA_DEG, R, label=f"e = {d_nm:g} nm")

        # CSV (two columns: R, angle_deg)
        fname = f"fig15_otto_e_{str(d_nm).replace('.', 'p')}nm.csv"
        save_curve_csv(os.path.join(out_dir, fname), THETA_DEG, R)

    ax.set_xlim(29, 40)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([30, 32, 34, 36, 38, 40])
    ax.set_yticks(np.arange(0.1, 1.0, 0.1))
    ax.set_xlabel("θ (°)")
    ax.set_ylabel("R")
    ax.set_title("Otto configuration (Fig. 15 replication)")
    ax.legend(loc="lower right")
    ax.grid(False)

    # ---------- Kretschmann configuration ----------
    # medium 2 = Ag (finite film), medium 3 = air (semi-infinite)
    ax = axes[1]
    for d_nm in KRETSCH_THICKNESSES_NM:
        R = reflectance_three_medium_p(
            THETA_DEG,
            eps1=EPS1,
            eps2=EPS_AG,
            eps3=EPS_AIR,
            d2_nm=d_nm,
            lambda0_nm=LAMBDA0_NM,
        )
        ax.plot(THETA_DEG, R, label=f"e = {d_nm:g} nm")

        # CSV (two columns: R, angle_deg)
        fname = f"fig15_kretschmann_e_{str(d_nm).replace('.', 'p')}nm.csv"
        save_curve_csv(os.path.join(out_dir, fname), THETA_DEG, R)

    ax.set_xlim(29, 40)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([30, 32, 34, 36, 38, 40])
    ax.set_yticks(np.arange(0.1, 1.0, 0.1))
    ax.set_xlabel("θ (°)")
    ax.set_ylabel("R")
    ax.set_title("Kretschmann configuration (Fig. 15 replication)")
    ax.legend(loc="lower right")
    ax.grid(False)

    # Optional: print minima for quick sanity-check against the paper
    print("=== Minima (sanity check) ===")
    for d_nm in OTTO_THICKNESSES_NM:
        R = reflectance_three_medium_p(
            THETA_DEG, EPS1, EPS_AIR, EPS_AG, d_nm, LAMBDA0_NM
        )
        i = np.argmin(R)
        print(
            f"Otto         e={d_nm:6.1f} nm : theta_min={THETA_DEG[i]:.4f} deg, Rmin={R[i]:.6g}"
        )

    for d_nm in KRETSCH_THICKNESSES_NM:
        R = reflectance_three_medium_p(
            THETA_DEG, EPS1, EPS_AG, EPS_AIR, d_nm, LAMBDA0_NM
        )
        i = np.argmin(R)
        print(
            f"Kretschmann  e={d_nm:6.1f} nm : theta_min={THETA_DEG[i]:.4f} deg, Rmin={R[i]:.6g}"
        )

    plt.show()
    print(f"\nCSV files saved in: {os.path.abspath(out_dir)}")
    print("Each CSV has exactly two columns: R,angle_deg")


if __name__ == "__main__":
    main()
