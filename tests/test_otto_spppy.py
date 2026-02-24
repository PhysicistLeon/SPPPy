import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SPPPy import ExperimentSPR, Layer, nm

LAMBDA0_NM = 546.1
EPS1 = 3.6168 + 0j
EPS_AIR = 1.0
EPS_AG = -10.9204 + 0.8334j
OTTO_THICKNESSES_NM = [424.8, 531.0, 637.2]
THETA_DEG = np.linspace(29.0, 40.0, 4001)


def eps_to_n(eps):
    if np.isscalar(eps):
        n = complex(np.sqrt(eps))
        if np.imag(n) < 0:
            n = -n
        return n
    else:
        n = np.sqrt(eps.astype(np.complex128))
        if np.imag(n) < 0:
            n = -n
        return n


def physical_sqrt(z):
    w = np.sqrt(z.astype(np.complex128))
    mask_flip = (np.imag(w) < -1e-14) | (
        (np.abs(np.imag(w)) <= 1e-14) & (np.real(w) < 0)
    )
    w[mask_flip] = -w[mask_flip]
    return w


def reflectance_three_medium_p(theta_deg, eps1, eps2, eps3, d2_nm, lambda0_nm):
    theta = np.deg2rad(theta_deg)
    u = np.sqrt(eps1) * np.sin(theta)
    w1 = physical_sqrt(eps1 - u**2)
    w2 = physical_sqrt(eps2 - u**2)
    w3 = physical_sqrt(eps3 - u**2)
    r12 = (eps2 * w1 - eps1 * w2) / (eps2 * w1 + eps1 * w2)
    r23 = (eps3 * w2 - eps2 * w3) / (eps3 * w2 + eps2 * w3)
    k0 = 2.0 * np.pi / lambda0_nm
    phase = np.exp(2j * k0 * w2 * d2_nm)
    r = (r12 + r23 * phase) / (1.0 + r12 * r23 * phase)
    R = np.abs(r) ** 2
    return R.real


n_prism = eps_to_n(EPS1)
n_air = float(np.sqrt(EPS_AIR))
n_ag = eps_to_n(EPS_AG)


@pytest.mark.parametrize("d_nm", OTTO_THICKNESSES_NM)
def test_otto_spppy(d_nm):
    R_ref = reflectance_three_medium_p(
        THETA_DEG, EPS1, EPS_AIR, EPS_AG, d_nm, LAMBDA0_NM
    )
    ref_theta, ref_R = THETA_DEG[np.argmin(R_ref)], np.min(R_ref)

    exp = ExperimentSPR(polarization="p")
    exp.wavelength = LAMBDA0_NM * nm
    exp.add(Layer(n_prism, 0, "Prism"))
    exp.add(Layer(n_air, d_nm * nm, "Air gap"))
    exp.add(Layer(n_ag, 0, "Ag substrate"))

    R_sppy = np.array(exp.R(angle_range=THETA_DEG))
    sppy_theta, sppy_R = THETA_DEG[np.argmin(R_sppy)], np.min(R_sppy)

    assert abs(ref_theta - sppy_theta) < 0.0001, f"theta mismatch at d={d_nm}"
    assert abs(ref_R - sppy_R) < 0.0001, f"R mismatch at d={d_nm}"
