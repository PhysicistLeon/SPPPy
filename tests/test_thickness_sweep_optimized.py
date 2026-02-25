import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SPPPy import ExperimentSPR, Layer, MaterialDispersion, nm

BASE_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "PermittivitiesBase.csv")


def _build_exp():
    exp = ExperimentSPR(polarization="p")
    exp.add(Layer(MaterialDispersion("BK7", base_file=BASE_CSV), 0, "Prism"))
    exp.add(Layer(MaterialDispersion("Ag", base_file=BASE_CSV), 55 * nm, "Ag"))
    exp.add(Layer(MaterialDispersion("SiO2", base_file=BASE_CSV), 0 * nm, "SiO2"))
    exp.add(Layer(1.0, 0, "Air"))
    return exp


def test_r_vs_thickness_matches_loop_for_reflectance():
    exp = _build_exp()
    theta = 62.0
    wl = 633.0 * nm
    thicknesses = np.linspace(0, 100, 11) * nm

    fast = np.array(exp.R_vs_thickness(2, thicknesses, theta=theta, wl=wl))

    exp2 = _build_exp()
    ref = []
    for h in thicknesses:
        exp2.layers[2].thickness = float(h)
        ref.append(np.abs(exp2.R_deg(theta=theta, wl=wl)) ** 2)
    ref = np.array(ref)

    np.testing.assert_allclose(fast, ref, rtol=1e-12, atol=1e-14)


def test_r_vs_thickness_matches_loop_for_complex_r():
    exp = _build_exp()
    theta = 50.0
    wl = 532.0 * nm
    thicknesses = np.linspace(5, 50, 8) * nm

    fast = np.array(exp.R_vs_thickness(1, thicknesses, theta=theta, wl=wl, is_complex=True))

    exp2 = _build_exp()
    ref = []
    for h in thicknesses:
        exp2.layers[1].thickness = float(h)
        ref.append(exp2.R_deg(theta=theta, wl=wl))
    ref = np.array(ref)

    np.testing.assert_allclose(fast, ref, rtol=1e-12, atol=1e-14)


def test_r_lambda_vs_thickness_matches_nested_loop():
    exp = _build_exp()
    theta = 62.0
    thicknesses = np.linspace(0, 100, 11) * nm
    wl_range = np.linspace(450, 650, 9) * nm

    fast = np.array(exp.R_lambda_vs_thickness(2, thicknesses, wl_range, theta=theta))

    exp2 = _build_exp()
    ref = []
    for h in thicknesses:
        exp2.layers[2].thickness = float(h)
        curve = [np.abs(exp2.R_deg(theta=theta, wl=float(wl))) ** 2 for wl in wl_range]
        ref.append(curve)
    ref = np.array(ref)

    np.testing.assert_allclose(fast, ref, rtol=1e-12, atol=1e-14)


def test_r_theta_vs_thickness_matches_nested_loop():
    exp = _build_exp()
    wl = 633.0 * nm
    thicknesses = np.linspace(0, 100, 9) * nm
    theta_range = np.linspace(45, 70, 13)

    fast = np.array(exp.R_theta_vs_thickness(2, thicknesses, theta_range, wl=wl))

    exp2 = _build_exp()
    ref = []
    for h in thicknesses:
        exp2.layers[2].thickness = float(h)
        curve = [np.abs(exp2.R_deg(theta=float(theta), wl=wl)) ** 2 for theta in theta_range]
        ref.append(curve)
    ref = np.array(ref)

    np.testing.assert_allclose(fast, ref, rtol=1e-12, atol=1e-14)
