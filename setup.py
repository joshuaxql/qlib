"""Setuptools hooks for platform wheels containing the ctypes C kernels."""

import importlib.util
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel

_helper_path = Path(__file__).resolve().with_name("build_native.py")
_helper_spec = importlib.util.spec_from_file_location("_qlib_build_native", _helper_path)
if _helper_spec is None or _helper_spec.loader is None:
    raise RuntimeError(f"Cannot load native build helper: {_helper_path}")
_helper = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helper)
LIBRARIES = _helper.LIBRARIES
build_libraries = _helper.build_libraries


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        # ctypes libraries have no Python ABI, but the wheel is platform-specific.
        return True


class NativeBuildPy(build_py):
    def run(self):
        super().run()
        directory = Path(self.build_lib) / "qlib" / "data" / "_libs"
        # A reused build directory must not carry binaries for another platform
        # into a wheel.  Source-tree binaries are excluded by package-data rules.
        for name in LIBRARIES:
            for suffix in (".dll", ".so", ".dylib", ".pyd"):
                stale = directory / (name + suffix)
                if stale.is_file():
                    stale.unlink()
        build_libraries(Path(__file__).resolve().parent / "qlib" / "data" / "_libs", directory)


class NativeBdistWheel(bdist_wheel):
    def get_tag(self):
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


setup(
    distclass=NativeDistribution,
    cmdclass={"build_py": NativeBuildPy, "bdist_wheel": NativeBdistWheel},
)
