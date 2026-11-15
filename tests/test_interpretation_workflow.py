"""Exercise the reproducible report, including every energetic-region cell."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def interpretation_output(tmp_path_factory):
    output = tmp_path_factory.mktemp("interpretation")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "interpretation.py"),
            "--no-show",
            "--output-dir",
            str(output),
        ],
        cwd=root,
        env={**os.environ, "MPLBACKEND": "Agg"},
        check=True,
        capture_output=True,
        text=True,
    )
    return output, json.loads((output / "interpretation.json").read_text())


def test_every_strong_region_accounted_and_independent_checks(interpretation_output):
    output, report = interpretation_output
    anchors = []
    for name, model in report["models"].items():
        assert (output / f"interpretation_{name}_stack_ray.png").exists()
        for variant, data in model["variants"].items():
            assert "reflections" not in data["bounds"]
            assert sum(r["cells"] for r in data["regions"]) == data["strong_cells"]
            assert data["accounting_error"] < 2e-12
            anchors.extend(data["checks"])
            with np.load(output / f"interpretation_{name}_{variant}_ray.npz") as cells:
                assert cells["strong"].sum() == data["strong_cells"]
                assert cells["group_energy"].shape[1] == data["strong_cells"]
                assert np.all(cells["group_energy"].sum(axis=0) > 0)
                assert np.all(cells["dominant_event_fold"] > 0)
                assert np.all(cells["fold"][cells["strong"]] >= 13)
            for row in data["groups"]:
                assert row["independent_offset_error_m"] < 1e-5
                assert row["independent_time_error_s"] < 1e-8
                if "fit" in row:
                    assert row["fit"]["fold"] == row["usable_fold"]
                    terms = row["budget_terms_s"]
                    assert row["timing_budget_s"] == pytest.approx(sum(terms.values()))
    # Six primaries + four uncontaminated water-train surface orders. Do not
    # let an empty anchor list silently pass the workflow's checks.
    assert len(anchors) == 10
    assert all(c["supported_ridge_match"] for c in anchors)


def test_interference_support_and_finite_spread_are_not_confused(interpretation_output):
    _, report = interpretation_output
    t2 = report["models"]["water_bottom"]["variants"]
    t3 = report["models"]["thick_5layer"]["variants"]
    assert t3["full"]["mixed_cells"] > 0
    assert any(r["cross_energy_fraction_range"][0] < -1 for r in t3["full"]["regions"])
    assert any(r["adjacent_fold_change_cells"] > 0 for r in t3["full"]["regions"])
    for row in t2["full"]["groups"]:
        if row["id"].endswith("-0") and "fit" in row:
            assert row["fit"]["velocity_m_s"] == pytest.approx(1500, abs=1e-5)
    p2 = next(r for r in t2["primaries"]["groups"] if r["id"] == "G2-2")
    assert p2["fit"]["velocity_m_s"] > p2["rms_m_s"] + 5
    assert p2["fit"]["max_residual_s"] < p2["timing_budget_s"]
    # Same model primary, different usable spread in full due to unsupported
    # neighbouring-event windows; this is not a different physical RMS.
    full_p2 = next(r for r in t2["full"]["groups"] if r["id"] == "G2-2")
    assert full_p2["usable_fold"] < p2["usable_fold"]
    assert full_p2["rms_m_s"] == p2["rms_m_s"]
    assert abs(full_p2["fit"]["velocity_m_s"] - p2["fit"]["velocity_m_s"]) > 1
