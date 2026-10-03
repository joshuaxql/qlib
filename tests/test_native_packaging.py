"""Native build safety and portability, without invoking a host compiler."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import build_native


class NativeBuildTests(unittest.TestCase):
    def test_setup_metadata_with_isolated_python_and_no_compiler(self):
        repository = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["QLIB_CC"] = "missing-qlib-test-compiler"
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-I", str(repository / "setup.py"), "egg_info", "--egg-base", temporary],
                cwd=repository,
                env=environment,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((Path(temporary) / "qlib_joshuaxql.egg-info" / "PKG-INFO").is_file())

    def test_platform_suffix_and_unsupported_platform(self):
        for platform_name, suffix in (("win32", ".dll"), ("linux", ".so"), ("darwin", ".dylib")):
            with self.subTest(platform=platform_name):
                self.assertEqual(build_native.library_suffix(platform_name), suffix)
        with self.assertRaisesRegex(build_native.NativeBuildError, "not supported"):
            build_native.library_suffix("freebsd14")

    def test_missing_compiler_fails_instead_of_using_checkout_binary(self):
        with patch("build_native.shutil.which", return_value=None):
            with self.assertRaisesRegex(build_native.NativeBuildError, "cannot be built"):
                build_native._compiler_command("missing-qlib-test-compiler", "win32")

    def test_windows_rejects_msvc_and_wrong_architecture(self):
        cases = (
            (["x86_64-pc-windows-msvc", "Microsoft compiler"], "MinGW-w64"),
            (["i686-w64-mingw32", "gcc 14.2"], "does not match"),
        )
        for outputs, error in cases:
            with (
                self.subTest(error=error),
                patch("build_native.struct.calcsize", return_value=8),
                patch("build_native.platform.machine", return_value="AMD64"),
            ):
                with patch("build_native.subprocess.check_output", side_effect=outputs):
                    with self.assertRaisesRegex(build_native.NativeBuildError, error):
                        build_native._compiler_identity(["compiler"], "win32")

    def test_linux_rejects_compiler_for_another_architecture(self):
        with (
            patch("build_native.platform.machine", return_value="x86_64"),
            patch("build_native.struct.calcsize", return_value=8),
            patch("build_native.subprocess.check_output", side_effect=["aarch64-linux-gnu", "gcc 14.2"]),
        ):
            with self.assertRaisesRegex(build_native.NativeBuildError, "does not match"):
                build_native._compiler_identity(["compiler"], "linux")

    def test_windows_statically_links_gcc_runtime_and_preserves_float_rules(self):
        command = build_native.compile_command(["gcc"], Path("rolling.c"), Path("rolling.dll"), "win32")
        self.assertIn("-static-libgcc", command)
        self.assertIn("-fno-fast-math", command)
        self.assertIn("-ffp-contract=off", command)
        self.assertNotIn("-march=native", command)

    def test_linux_shared_library_resolves_all_symbols(self):
        command = build_native.compile_command(["gcc"], Path("rolling.c"), Path("rolling.so"), "linux")
        self.assertIn("-fPIC", command)
        self.assertIn("-shared", command)
        self.assertIn("-Wl,--no-undefined", command)
        self.assertEqual(command[-1], "-lm")

    def test_macos_honors_architecture_and_deployment_target(self):
        with patch.dict(os.environ, {"ARCHFLAGS": "-arch arm64", "MACOSX_DEPLOYMENT_TARGET": "11.0"}):
            command = build_native.compile_command(["clang"], Path("pit.c"), Path("pit.dylib"), "darwin")
        self.assertIn("-dynamiclib", command)
        self.assertIn("-Wl,-install_name,@rpath/pit.dylib", command)
        self.assertEqual(command[command.index("-arch") + 1], "arm64")
        self.assertIn("-mmacosx-version-min=11.0", command)

    def test_every_library_is_freshly_compiled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "src", root / "out"
            source.mkdir()
            for name in build_native.LIBRARIES:
                (source / (name + ".c")).write_text("/* source */", encoding="utf-8")
                (source / (name + ".dll")).write_bytes(b"stale checkout DLL")
            calls = []

            def compile_to_output(command, check):
                self.assertTrue(check)
                calls.append(command)
                Path(command[command.index("-o") + 1]).write_bytes(b"freshly compiled")

            with patch("build_native._compiler_command", return_value=["gcc"]):
                with patch("build_native._compiler_identity", return_value=("x86_64-w64-mingw32", "gcc")):
                    with patch("build_native.subprocess.run", side_effect=compile_to_output):
                        libraries = build_native.build_libraries(source, output, platform_name="win32")
            self.assertEqual(len(calls), 3)
            self.assertEqual({item.name for item in libraries}, {name + ".dll" for name in build_native.LIBRARIES})
            self.assertTrue(all(item.read_bytes() == b"freshly compiled" for item in libraries))
            self.assertFalse(list(output.glob(".qlib-native-*")))

    def test_failed_compile_does_not_publish_partial_libraries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "src", root / "out"
            source.mkdir()
            output.mkdir()
            for name in build_native.LIBRARIES:
                (source / (name + ".c")).write_text("/* source */", encoding="utf-8")
                (output / (name + ".dll")).write_bytes(b"previous complete build")
            with patch("build_native._compiler_command", return_value=["gcc"]):
                with patch("build_native._compiler_identity", return_value=("target", "gcc")):
                    with patch("build_native.subprocess.run", side_effect=subprocess.CalledProcessError(1, ["gcc"])):
                        with self.assertRaisesRegex(build_native.NativeBuildError, "Failed to build rolling"):
                            build_native.build_libraries(source, output, platform_name="win32")
            self.assertTrue(all(item.read_bytes() == b"previous complete build" for item in output.glob("*.dll")))
            self.assertFalse(list(output.glob(".qlib-native-*")))


if __name__ == "__main__":
    unittest.main()
