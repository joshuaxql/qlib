"""Native wheel paths, lazy loading and the source-install fallback."""

from importlib.machinery import ExtensionFileLoader, FileFinder, SourceFileLoader, SOURCE_SUFFIXES
from importlib.util import module_from_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qlib.data import ops
from qlib.data._libs import _path, expanding, pit, rolling


class NativeLoadingTest(unittest.TestCase):
    def tearDown(self):
        for module in (rolling, expanding, pit):
            module._library.cache_clear()

    def test_platform_paths_stay_in_the_package(self):
        for platform, prefix, suffix in (("win32", "", ".dll"), ("linux", "lib", ".so"),
                                         ("darwin", "", ".dylib")):
            with self.subTest(platform=platform), patch.object(_path.sys, "platform", platform):
                for name in ("rolling", "expanding", "pit"):
                    self.assertEqual(_path.library_path(name), Path(_path.__file__).with_name(prefix + name + suffix))

    def test_linux_ctypes_libraries_do_not_shadow_python_wrappers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("rolling", "expanding", "pit"):
                (root / (name + ".py")).write_text("wrapper_loaded = True\n", encoding="utf-8")
                (root / ("lib" + name + ".so")).write_bytes(b"ctypes-only shared library")
            # Use Linux's real import search order, even on a Windows test host.
            finder = FileFinder(directory, (ExtensionFileLoader, [".so"]),
                                (SourceFileLoader, SOURCE_SUFFIXES))
            for name in ("rolling", "expanding", "pit"):
                with self.subTest(name=name):
                    spec = finder.find_spec(name)
                    self.assertIsInstance(spec.loader, SourceFileLoader)
                    module = module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.assertTrue(module.wrapper_loaded)
            # The old filename wins over its wrapper and would require PyInit_*.
            (root / "expanding.so").write_bytes(b"ctypes-only shared library")
            finder.invalidate_caches()
            self.assertIsInstance(finder.find_spec("expanding").loader, ExtensionFileLoader)

    def test_missing_libraries_are_lazy_and_have_a_working_fallback(self):
        values = pd.Series([1.0, np.nan, 3.0, 5.0])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(rolling, "LIBRARY_PATH", root / "rolling.dll"), \
                    patch.object(expanding, "LIBRARY_PATH", root / "expanding.dll"), \
                    patch.object(pit, "LIBRARY_PATH", root / "pit.dll"), \
                    patch("ctypes.CDLL", side_effect=AssertionError("Missing libraries must not be loaded")):
                for module in (rolling, expanding, pit):
                    self.assertFalse(module.is_available())
                    with self.assertRaisesRegex(ImportError, "install a platform wheel"):
                        module._library()
                self.assertFalse(rolling.supports("Mean"))
                self.assertFalse(expanding.supports("Mean"))
                np.testing.assert_allclose(ops.OPERATORS["Mean"](values, 2), [1, 1, 3, 4])
                np.testing.assert_allclose(ops.OPERATORS["Mean"](values, 0), [1, 1, 2, 3])
                np.testing.assert_array_equal(pit.asof_indices([2, 5, 5, 8], [0, 2, 4], [2, 2, 0], 5),
                                              [1, 2, -1])

    def test_broken_installed_libraries_do_not_silently_fall_back(self):
        with TemporaryDirectory() as directory:
            broken = Path(directory) / "broken-library"
            broken.write_bytes(b"not a shared library")
            for module in (rolling, expanding, pit):
                with self.subTest(module=module.__name__), patch.object(module, "LIBRARY_PATH", broken), \
                        patch("ctypes.CDLL", side_effect=OSError("wrong architecture")):
                    module._library.cache_clear()
                    self.assertTrue(module.is_available())
                    with self.assertRaisesRegex(OSError, "wrong architecture"):
                        module._library()


if __name__ == "__main__":
    unittest.main()
