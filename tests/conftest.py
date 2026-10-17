"""Shared test configuration.

The cloud container cannot install the Mac-only perception/narrator stack, so a test that fails only
because one of those packages is missing is reported as skipped (with the module name), not failed.
"""

import pytest

MAC_ONLY = {"ultralytics", "sounddevice", "vosk", "mlx", "mlx_lm", "torch"}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    excinfo = outcome.excinfo
    if excinfo and issubclass(excinfo[0], ModuleNotFoundError):
        name = (excinfo[1].name or "").split(".")[0]
        if name in MAC_ONLY:
            outcome.force_exception(pytest.skip.Exception(f"Mac-only dependency '{name}' not installed"))
