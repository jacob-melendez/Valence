"""Shared test configuration.

The cloud container cannot install the Mac-only perception/narrator stack (torch's index is blocked
there, mlx is Apple-only). So that modules importing those packages at top level can still be collected,
a placeholder module is installed for each missing one; any real use of it raises ModuleNotFoundError,
and a test that fails only for that reason is reported as skipped (with the module name), not failed.
On the Mac every package is installed and none of this activates.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types

import pytest

MAC_ONLY = {"ultralytics", "sounddevice", "vosk", "mlx", "mlx_lm", "torch", "torchvision", "cv2"}
MISSING = {name for name in MAC_ONLY if importlib.util.find_spec(name) is None}


class _Missing(types.ModuleType):
    def __getattr__(self, attr):
        if attr.startswith("__") and attr != "__file__":
            raise AttributeError(attr)  # import machinery and pytest probe dunders
        root = self.__name__.split(".")[0]
        raise ModuleNotFoundError(f"No module named '{root}'", name=root)


class _MissingFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in MISSING:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        mod = _Missing(spec.name)
        mod.__path__ = []
        return mod

    def exec_module(self, module):
        pass


if MISSING:
    sys.meta_path.insert(0, _MissingFinder())


def _skip_if_mac_only(outcome):
    excinfo = outcome.excinfo
    if excinfo and issubclass(excinfo[0], (ModuleNotFoundError, AttributeError)):
        name = (getattr(excinfo[1], "name", None) or "").split(".")[0]
        if name in MAC_ONLY:
            outcome.force_exception(pytest.skip.Exception(f"Mac-only dependency '{name}' not installed"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):  # fixtures that touch a Mac-only package
    _skip_if_mac_only((yield))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _skip_if_mac_only((yield))
