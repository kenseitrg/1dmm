"""Loader, validation, depth-merging and blocking tests."""

import numpy as np
import pytest

from mm1d.io import (
    Log,
    LogError,
    block_log_uniform_twt,
    load_blocked_model,
    load_log,
    load_log_pair,
)

THREE_COLUMN = """# comment line
depth_m,vp_m_s,rho_kg_m3
0,1500,1000
300,2400,2100
900,3000,2300
"""


def write_file(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# -- three-column sampled log -------------------------------------------------


def test_load_three_column_csv_with_header(tmp_path):
    log = load_log(write_file(tmp_path, "log.csv", THREE_COLUMN))
    np.testing.assert_allclose(log.depth_m, [0.0, 300.0, 900.0])
    np.testing.assert_allclose(log.vp_m_s, [1500.0, 2400.0, 3000.0])
    np.testing.assert_allclose(log.rho_kg_m3, [1000.0, 2100.0, 2300.0])
    assert "Log" in repr(log)


def test_load_whitespace_separated_no_header(tmp_path):
    text = "0 1500 1000\n300 2400 2100\n900 3000 2300\n"
    log = load_log(write_file(tmp_path, "log.txt", text))
    np.testing.assert_allclose(log.depth_m, [0.0, 300.0, 900.0])


def test_load_rejects_wrong_column_count(tmp_path):
    with pytest.raises(LogError, match="expected 3 columns"):
        load_log(write_file(tmp_path, "bad.csv", "0 1500\n300 2400\n"))


def test_load_rejects_inconsistent_columns(tmp_path):
    text = "depth,vp,rho\n0,1500,1000\n300,2400,2100,0\n"
    with pytest.raises(LogError, match="inconsistent column counts"):
        load_log(write_file(tmp_path, "bad.csv", text))


def test_load_rejects_non_monotonic_depth(tmp_path):
    text = "0,1500,1000\n900,3000,2300\n300,2400,2100\n"
    with pytest.raises(LogError, match="strictly increasing"):
        load_log(write_file(tmp_path, "bad.csv", text))


def test_load_rejects_nan(tmp_path):
    text = "0,1500,1000\n300,nan,2100\n"
    with pytest.raises(LogError, match="non-finite vp_m_s"):
        load_log(write_file(tmp_path, "bad.csv", text))


def test_load_rejects_nonpositive_values(tmp_path):
    text = "0,1500,1000\n300,-2400,2100\n"
    with pytest.raises(LogError, match="vp_m_s must be > 0"):
        load_log(write_file(tmp_path, "bad.csv", text))
    text = "0,1500,1000\n300,2400,0\n"
    with pytest.raises(LogError, match="rho_kg_m3 must be > 0"):
        load_log(write_file(tmp_path, "bad.csv", text))


def test_load_rejects_unparseable_data_line(tmp_path):
    text = "0,1500,1000\n300,??,2100\n900,3000,2300\n"
    with pytest.raises(LogError, match=r"could not parse line\(s\) 2"):
        load_log(write_file(tmp_path, "bad.csv", text))


def test_load_rejects_empty_file(tmp_path):
    with pytest.raises(LogError, match="no numeric data rows"):
        load_log(write_file(tmp_path, "empty.csv", "# only comments\n\n"))


def test_log_record_validates_itself():
    with pytest.raises(LogError, match="strictly increasing"):
        Log(np.array([0.0, 100.0, 50.0]), np.full(3, 2000.0), np.full(3, 2000.0))
    with pytest.raises(LogError, match="equal length"):
        Log(np.array([0.0, 100.0]), np.full(2, 2000.0), np.full(3, 2000.0))


# -- two-file merge on depth ----------------------------------------------------


def test_pair_merge_exact(tmp_path):
    vp_path = write_file(tmp_path, "vp.csv", "0,1500\n300,2400\n900,3000\n")
    rho_path = write_file(tmp_path, "rho.csv", "0,1000\n300,2100\n900,2300\n")
    log = load_log_pair(vp_path, rho_path)
    np.testing.assert_allclose(log.depth_m, [0.0, 300.0, 900.0])
    np.testing.assert_allclose(log.rho_kg_m3, [1000.0, 2100.0, 2300.0])


def test_pair_merge_within_tolerance(tmp_path):
    vp_path = write_file(tmp_path, "vp.csv", "0,1500\n300,2400\n900,3000\n")
    rho_path = write_file(tmp_path, "rho.csv", "0.2,1000\n300.2,2100\n900.2,2300\n")
    log = load_log_pair(vp_path, rho_path, depth_tolerance_m=0.5)
    np.testing.assert_allclose(log.depth_m, [0.1, 300.1, 900.1])
    np.testing.assert_allclose(log.rho_kg_m3, [1000.0, 2100.0, 2300.0])


def test_pair_merge_beyond_tolerance_fails(tmp_path):
    vp_path = write_file(tmp_path, "vp.csv", "0,1500\n300,2400\n900,3000\n")
    rho_path = write_file(tmp_path, "rho.csv", "2,1000\n302,2100\n902,2300\n")
    with pytest.raises(LogError, match="tolerance"):
        load_log_pair(vp_path, rho_path, depth_tolerance_m=0.5)


def test_pair_rejects_wrong_columns(tmp_path):
    vp_path = write_file(tmp_path, "vp.csv", "0,1500,1000\n300,2400,2100\n")
    rho_path = write_file(tmp_path, "rho.csv", "0,1000\n300,2100\n")
    with pytest.raises(LogError, match="expected 2 columns"):
        load_log_pair(vp_path, rho_path)


# -- blocked models ---------------------------------------------------------------


def test_blocked_model_loader(tmp_path):
    text = "# canonical T2\ndepth_m,vp_m_s,rho_kg_m3\n0,1500,1000\n300,2400,2100\n900,3000,2300\n"
    model = load_blocked_model(write_file(tmp_path, "water_bottom.csv", text))
    assert model.name == "water_bottom"
    assert model.n_layers == 3
    np.testing.assert_allclose(model.h_m, [300.0, 600.0])
    np.testing.assert_allclose(model.vp_m_s, [1500.0, 2400.0, 3000.0])
    np.testing.assert_allclose(model.rho_kg_m3, [1000.0, 2100.0, 2300.0])


def test_blocked_model_requires_depth_zero_start(tmp_path):
    text = "100,1500,1000\n400,2400,2100\n"
    with pytest.raises(LogError, match="depth 0"):
        load_blocked_model(write_file(tmp_path, "bad.csv", text))


def test_blocked_model_requires_two_rows(tmp_path):
    with pytest.raises(LogError, match="at least 2 rows"):
        load_blocked_model(write_file(tmp_path, "bad.csv", "0,1500,1000\n"))


def test_canonical_data_files_load(data_dir):
    for name in ("single_interface", "water_bottom", "thick_5layer"):
        model = load_blocked_model(data_dir / f"{name}.csv")
        assert model.n_layers >= 2
        assert model.name == name


# -- uniform-TWT blocking ----------------------------------------------------------


def test_block_constant_log_round_trips_exactly():
    depth = np.arange(0.0, 1001.0, 10.0)
    log = Log(depth, np.full_like(depth, 2000.0), np.full_like(depth, 2200.0))
    model = block_log_uniform_twt(log, layer_twt_s=0.1)
    assert len(model.h_m) == 10  # 1.0 s total TWT / 0.1 s per layer
    np.testing.assert_allclose(model.h_m, 100.0)
    np.testing.assert_allclose(model.vp_m_s, 2000.0)
    np.testing.assert_allclose(model.rho_kg_m3, 2200.0)
    np.testing.assert_allclose(model.layer_twt_s, 0.1)
    assert model.total_depth_m == pytest.approx(1000.0)
    # half-space duplicates the deepest cell -> invisible base interface
    assert model.reflections_normal[-1] == 0.0


def test_block_gradient_log_preserves_invariants():
    depth = np.linspace(0.0, 2000.0, 401)
    vp = 1500.0 + 1.5 * depth
    rho = 2000.0 + 0.1 * depth
    model = block_log_uniform_twt(Log(depth, vp, rho), layer_twt_s=0.2)
    ow_total = np.log(4500.0 / 1500.0) / 1.5  # closed form for linear v(z)
    twt_total = 2.0 * ow_total
    assert len(model.h_m) == 7  # 6 x 0.2 s cells + deepest cell absorbing the remainder
    np.testing.assert_allclose(model.layer_twt_s[:-1], 0.2)
    assert model.layer_twt_s[-1] == pytest.approx(twt_total - 6 * 0.2, rel=1e-12)
    # no sliver layers: every layer is at least the target TWT thick
    assert model.layer_twt_s.min() >= 0.2 - 1e-12
    assert (model.h_m / model.vp_m_s[:-1]).sum() == pytest.approx(ow_total, rel=1e-12)
    assert model.h_m @ model.rho_kg_m3[:-1] == pytest.approx(np.trapezoid(rho, depth), rel=1e-12)
    assert model.total_depth_m == pytest.approx(2000.0)
    # interval velocities stay within the logged velocity range
    assert model.vp_m_s.min() >= vp.min()
    assert model.vp_m_s.max() <= vp.max()


def test_block_max_layers_cap():
    depth = np.linspace(0.0, 2000.0, 401)
    vp = 1500.0 + 1.5 * depth
    rho = 2000.0 + 0.1 * depth
    model = block_log_uniform_twt(Log(depth, vp, rho), layer_twt_s=0.05, max_layers=6)
    assert len(model.h_m) == 6
    np.testing.assert_allclose(model.layer_twt_s, model.layer_twt_s[0])


def test_block_layer_twt_larger_than_total():
    depth = np.arange(0.0, 1001.0, 10.0)
    log = Log(depth, np.full_like(depth, 2000.0), np.full_like(depth, 2200.0))
    model = block_log_uniform_twt(log, layer_twt_s=5.0)
    assert len(model.h_m) == 1
    np.testing.assert_allclose(model.h_m, 1000.0)
    np.testing.assert_allclose(model.vp_m_s, 2000.0)


def test_block_rejects_invalid_arguments():
    depth = np.arange(0.0, 101.0, 10.0)
    log = Log(depth, np.full_like(depth, 2000.0), np.full_like(depth, 2000.0))
    with pytest.raises(LogError, match="layer_twt_s"):
        block_log_uniform_twt(log, layer_twt_s=0.0)
    with pytest.raises(LogError, match="max_layers"):
        block_log_uniform_twt(log, layer_twt_s=0.1, max_layers=0)
