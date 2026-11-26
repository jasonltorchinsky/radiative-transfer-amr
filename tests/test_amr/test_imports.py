"""
Regression test for issue #6: the error-indicator package must be importable
under the lower-case name used by every import, also on case-sensitive file systems.
"""

# Standard Library Imports
import importlib
import pathlib

# Local Library Imports
import amr


def test_error_indicator_package_name_matches_imports():
    package_dir = pathlib.Path(amr.__file__).parent if amr.__file__ else pathlib.Path(list(amr.__path__)[0])
    names = {path.name for path in package_dir.iterdir() if path.is_dir()}
    assert "error_indicator" in names
    assert "Error_Indicator" not in names


def test_error_indicator_imports():
    module = importlib.import_module("amr.error_indicator")
    assert hasattr(module, "Error_Indicator")
    importlib.import_module("amr.error_indicator.ref_by_ind")
