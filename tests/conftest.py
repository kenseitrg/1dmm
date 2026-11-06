"""Shared fixtures: canonical models T1-T3 from data/."""

from pathlib import Path

import pytest

from mm1d.io import load_blocked_model

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Path to the canonical model directory."""
    return DATA_DIR


@pytest.fixture(scope="session")
def t1(data_dir: Path):
    """T1 single-interface model: one 500 m layer over a half-space."""
    return load_blocked_model(data_dir / "single_interface.csv")


@pytest.fixture(scope="session")
def t2(data_dir: Path):
    """T2 water-bottom model: water layer + sediment over a half-space."""
    return load_blocked_model(data_dir / "water_bottom.csv")


@pytest.fixture(scope="session")
def t3(data_dir: Path):
    """T3 thick 5-layer model with a soft interbed."""
    return load_blocked_model(data_dir / "thick_5layer.csv")


@pytest.fixture(scope="session")
def canonical_models(t1, t2, t3) -> dict:
    """All canonical models keyed by name (for parametrized physics tests)."""
    return {"t1": t1, "t2": t2, "t3": t3}
