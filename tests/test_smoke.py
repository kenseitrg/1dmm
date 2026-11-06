"""Smoke tests: package imports, metadata, and the shared variant enum."""

import importlib

import mm1d

SUBMODULES = [
    "io",
    "model",
    "reflectivity",
    "wavelet",
    "gathers",
    "semblance",
    "events",
    "viz",
    "units",
]


def test_version():
    assert mm1d.__version__ == "0.1.0"


def test_submodules_importable():
    for name in SUBMODULES:
        assert importlib.import_module(f"mm1d.{name}") is not None


def test_wavefield_variants():
    from mm1d.reflectivity import ALL_VARIANTS

    assert set(ALL_VARIANTS) == {"full", "no_surface", "primaries"}
