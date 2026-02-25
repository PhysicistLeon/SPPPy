import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from examples.find_minima import find_resonance_wavelength

from SPPPy import ExperimentSPR, Layer, MaterialDispersion, nm

BASE_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "PermittivitiesBase.csv"
)


def _build_exp():
    exp = ExperimentSPR(polarization="p")
    exp.add(Layer(MaterialDispersion("BK7", base_file=BASE_CSV), 0, "Prism"))
    exp.add(Layer(MaterialDispersion("Ag", base_file=BASE_CSV), 55 * nm, "Ag"))
    exp.add(Layer(MaterialDispersion("SiO2", base_file=BASE_CSV), 10 * nm, "SiO2"))
    exp.add(Layer(1.0, 0, "Air"))
    return exp


def test_find_resonance_with_layer_index_matches_without():
    """Test that find_resonance_wavelength gives same result with and without layer_index."""
    exp_old = _build_exp()
    exp_new = _build_exp()

    result_old = find_resonance_wavelength(
        exp_old,
        angle_deg=60.0,
        wl_min=400 * nm,
        wl_max=800 * nm,
        step=2 * nm,
        lambda_guess=None,
        refine=True,
        verbose=False,
    )

    result_new = find_resonance_wavelength(
        exp_new,
        angle_deg=60.0,
        wl_min=400 * nm,
        wl_max=800 * nm,
        step=2 * nm,
        lambda_guess=None,
        refine=True,
        verbose=False,
        layer_index=2,
    )

    lambda_old, R_old, info_old = result_old
    lambda_new, R_new, info_new = result_new

    np.testing.assert_allclose(lambda_old, lambda_new, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(R_old, R_new, rtol=1e-10, atol=1e-12)


def test_find_resonance_with_layer_index_and_lambda_guess():
    """Test that branch tracking works correctly with layer_index."""
    exp_old = _build_exp()
    exp_new = _build_exp()

    lambda_guess = 550 * nm

    result_old = find_resonance_wavelength(
        exp_old,
        angle_deg=60.0,
        wl_min=400 * nm,
        wl_max=800 * nm,
        step=2 * nm,
        lambda_guess=lambda_guess,
        guess_window=50 * nm,
        refine=True,
        verbose=False,
    )

    result_new = find_resonance_wavelength(
        exp_new,
        angle_deg=60.0,
        wl_min=400 * nm,
        wl_max=800 * nm,
        step=2 * nm,
        lambda_guess=lambda_guess,
        guess_window=50 * nm,
        refine=True,
        verbose=False,
        layer_index=2,
    )

    lambda_old, R_old, _ = result_old
    lambda_new, R_new, _ = result_new

    np.testing.assert_allclose(lambda_old, lambda_new, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(R_old, R_new, rtol=1e-10, atol=1e-12)


def test_find_resonance_refine_step_with_layer_index():
    """Test that refine step works correctly with layer_index."""
    exp = _build_exp()

    result = find_resonance_wavelength(
        exp,
        angle_deg=55.0,
        wl_min=500 * nm,
        wl_max=700 * nm,
        step=5 * nm,
        refine=True,
        verbose=False,
        layer_index=2,
    )

    lambda_res, R_min, info = result

    assert 500 * nm <= lambda_res <= 700 * nm
    assert 0.0 <= R_min <= 1.0
    assert "used_range_nm" in info


def test_find_resonance_info_structure_with_layer_index():
    """Test that info dict has correct structure with layer_index."""
    exp = _build_exp()

    _, _, info = find_resonance_wavelength(
        exp,
        angle_deg=45.0,
        wl_min=400 * nm,
        wl_max=900 * nm,
        step=5 * nm,
        verbose=False,
        layer_index=2,
    )

    assert "lambda_res_nm" in info
    assert "angle_deg" in info
    assert "requested_range_nm" in info
    assert "used_range_nm" in info
    assert "range_clipped" in info
    assert "selection_mode" in info
    assert "coarse_step_nm" in info
    assert "materials_limits_nm" in info
