"""Analytic validation of the layered model and acoustic R/T coefficients."""

import numpy as np
import pytest

from mm1d.model import LayeredModel, ModelError, acoustic_reflection_p

ANGLES_DEG = np.arange(0.0, 40.0 + 1e-9, 1.0)
"""Default gather-angle grid from PLAN.md (0-40 deg, 1 deg step)."""


def r_normal(z_above: float, z_below: float) -> float:
    """Closed-form normal-incidence reflection coefficient."""
    return (z_below - z_above) / (z_below + z_above)


# -- normal incidence against hand-computed interface values -------------------


def test_t1_normal_incidence(t1):
    z1, z2 = 2000.0 * 2000.0, 3000.0 * 2500.0
    assert t1.reflection(0) == pytest.approx(r_normal(z1, z2), rel=1e-12)
    # PLAN.md hand value
    assert t1.reflection(0) == pytest.approx(0.304, abs=2e-3)


def test_t2_normal_incidence(t2):
    expected = [
        r_normal(1500.0 * 1000.0, 2400.0 * 2100.0),
        r_normal(2400.0 * 2100.0, 3000.0 * 2300.0),
    ]
    np.testing.assert_allclose(t2.reflections_normal, expected, rtol=1e-12)
    # PLAN.md hand values
    assert t2.reflection(0) == pytest.approx(0.541, abs=2e-3)
    assert t2.reflection(1) == pytest.approx(0.156, abs=2e-3)


def test_t3_normal_incidence(t3):
    impedances = t3.vp_m_s * t3.rho_kg_m3
    expected = [r_normal(impedances[i], impedances[i + 1]) for i in range(t3.n_interfaces)]
    np.testing.assert_allclose(t3.reflections_normal, expected, rtol=1e-12)
    # PLAN.md hand values (recomputed exactly from the model above; these are
    # rounded sanity anchors: +0.266, -0.113, +0.165, +0.132)
    np.testing.assert_allclose(t3.reflections_normal, [0.266, -0.113, 0.165, 0.132], atol=2e-3)


def test_normal_incidence_pressure_transmission(t2):
    pairs = [(1500.0 * 1000.0, 2400.0 * 2100.0), (2400.0 * 2100.0, 3000.0 * 2300.0)]
    for i, (za, zb) in enumerate(pairs):
        t_down, t_up = t2.transmission(i)
        assert t_down == pytest.approx(2.0 * zb / (za + zb), rel=1e-12)
        assert t_up == pytest.approx(2.0 * za / (za + zb), rel=1e-12)


# -- energy conservation and identities across the angle grid ------------------


@pytest.mark.parametrize("model_name", ["t1", "t2", "t3"])
def test_energy_conservation_vs_angle(canonical_models, model_name):
    model = canonical_models[model_name]
    for i in range(model.n_interfaces):
        r = model.reflection(i, ANGLES_DEG)
        t_hat_down, t_hat_up = model.energy_transmission(i, ANGLES_DEG)
        assert np.allclose(np.abs(r) ** 2 + np.abs(t_hat_down) ** 2, 1.0, atol=1e-12)
        assert np.allclose(np.abs(r) ** 2 + np.abs(t_hat_up) ** 2, 1.0, atol=1e-12)
        # energy-normalized coefficients are real-valued and direction-symmetric
        assert not np.iscomplexobj(t_hat_down)
        np.testing.assert_allclose(t_hat_down, t_hat_up, rtol=1e-12)


@pytest.mark.parametrize("model_name", ["t1", "t2", "t3"])
def test_round_trip_transmission_identity(canonical_models, model_name):
    model = canonical_models[model_name]
    for i in range(model.n_interfaces):
        r = model.reflection(i, ANGLES_DEG)
        t_down, t_up = model.transmission(i, ANGLES_DEG)
        np.testing.assert_allclose(t_down * t_up, 1.0 - r**2, rtol=1e-12, atol=1e-14)


def test_pressure_reflection_antisymmetry():
    # reflection from above equals minus reflection from below at the same p
    p = np.sin(np.radians(30.0)) / 2400.0
    r_down = acoustic_reflection_p(2400.0, 2100.0, 1500.0, 1000.0, p)
    r_up = acoustic_reflection_p(1500.0, 1000.0, 2400.0, 2100.0, p)
    assert r_down == pytest.approx(-r_up, rel=1e-12)


# -- critical angle behaviour ---------------------------------------------------


def test_post_critical_total_internal_reflection(t2):
    critical = t2.critical_angle_deg(0)
    assert critical == pytest.approx(np.degrees(np.arcsin(1500.0 / 2400.0)), rel=1e-12)
    r45 = t2.reflection(0, 45.0)
    assert isinstance(r45, complex)
    assert abs(r45) == pytest.approx(1.0, abs=1e-12)
    t_hat_down, t_hat_up = t2.energy_transmission(0, 45.0)
    assert t_hat_down == 0.0
    assert t_hat_up == 0.0
    # subcritical reflection is real and grows toward +1 with angle (v increases)
    r30 = t2.reflection(0, 30.0)
    assert isinstance(r30, float)
    assert r30 > t2.reflection(0)


def test_no_critical_angle_when_velocity_decreases(t3):
    assert t3.critical_angle_deg(1) is None
    assert t3.critical_angle_deg(0) == pytest.approx(np.degrees(np.arcsin(1800.0 / 2700.0)))


def test_slowness_above_incidence_limit_rejected():
    with pytest.raises(ModelError, match="incidence limit"):
        acoustic_reflection_p(1500.0, 1000.0, 2400.0, 2100.0, 1.0 / 1000.0)


# -- geometry -------------------------------------------------------------------


def test_geometry_t1(t1):
    assert t1.n_layers == 2
    assert t1.n_interfaces == 1
    np.testing.assert_allclose(t1.h_m, [500.0])
    np.testing.assert_allclose(t1.interface_depth_m, [500.0])
    np.testing.assert_allclose(t1.layer_twt_s, [0.5])
    np.testing.assert_allclose(t1.interface_twt_s, [0.5])
    assert t1.total_depth_m == pytest.approx(500.0)


def test_geometry_t2(t2):
    assert t2.n_layers == 3
    np.testing.assert_allclose(t2.h_m, [300.0, 600.0])
    np.testing.assert_allclose(t2.interface_depth_m, [300.0, 900.0])
    np.testing.assert_allclose(t2.layer_twt_s, [0.4, 0.5])
    np.testing.assert_allclose(t2.interface_twt_s, [0.4, 0.9])


def test_geometry_t3(t3):
    assert t3.n_layers == 5
    np.testing.assert_allclose(t3.h_m, [450.0, 675.0, 345.0, 750.0])
    np.testing.assert_allclose(t3.interface_depth_m, [450.0, 1125.0, 1470.0, 2220.0])
    np.testing.assert_allclose(t3.layer_twt_s, [0.5, 0.5, 0.3, 0.5])
    np.testing.assert_allclose(t3.interface_twt_s, [0.5, 1.0, 1.3, 1.8])


def test_scalar_in_scalar_out(t1):
    r = t1.reflection(0, 10.0)
    assert isinstance(r, float)
    r_arr = t1.reflection(0, np.array([10.0, 20.0]))
    assert r_arr.shape == (2,)
    assert r_arr[0] == pytest.approx(r, rel=1e-12)


# -- model validation -------------------------------------------------------------


def test_model_rejects_negative_thickness():
    with pytest.raises(ModelError, match="thickness"):
        LayeredModel(
            h_m=np.array([100.0, -5.0]),
            vp_m_s=np.array([1500.0, 2500.0, 3000.0]),
            rho_kg_m3=np.array([1000.0, 2100.0, 2300.0]),
        )


def test_model_rejects_bad_lengths():
    with pytest.raises(ModelError, match="n_layers - 1"):
        LayeredModel(
            h_m=np.array([100.0, 100.0]),
            vp_m_s=np.array([1500.0, 2500.0]),
            rho_kg_m3=np.array([1000.0, 2100.0]),
        )
    with pytest.raises(ModelError, match="equal length"):
        LayeredModel(
            h_m=np.array([100.0]),
            vp_m_s=np.array([1500.0, 2500.0]),
            rho_kg_m3=np.array([1000.0]),
        )


def test_model_rejects_nonpositive_velocity():
    with pytest.raises(ModelError, match="vp_m_s"):
        LayeredModel(
            h_m=np.array([100.0]),
            vp_m_s=np.array([1500.0, -2500.0]),
            rho_kg_m3=np.array([1000.0, 2100.0]),
        )


def test_model_rejects_single_layer():
    with pytest.raises(ModelError, match="at least 2 layers"):
        LayeredModel(h_m=np.array([]), vp_m_s=np.array([1500.0]), rho_kg_m3=np.array([1000.0]))


def test_interface_index_out_of_range(t1):
    with pytest.raises(ModelError, match="out of range"):
        t1.reflection(1)
    with pytest.raises(ModelError, match="out of range"):
        t1.critical_angle_deg(-1)


def test_summary_and_repr(t2):
    text = t2.summary()
    assert "water_bottom" in text
    assert "3 layers" in text
    assert "inf" in text
    assert "LayeredModel" in repr(t2)
