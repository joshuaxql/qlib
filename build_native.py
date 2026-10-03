"""Compile Qlib's Python-ABI-independent C libraries using only the stdlib.

This module is shared by the wheel builder and scripts/build_rolling.py.  It
must remain importable before the project's runtime dependencies are installed.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile


LIBRARIES = ("rolling", "expanding", "pit")
COMMON_FLAGS = (
    "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror", "-pedantic",
    "-fno-fast-math", "-ffp-contract=off",
)


class NativeBuildError(RuntimeError):
    """The native libraries could not be built for the current interpreter."""


def library_suffix(platform_name: str | None = None) -> str:
    name = sys.platform if platform_name is None else platform_name
    if name == "win32":
        return ".dll"
    if name == "darwin":
        return ".dylib"
    if name.startswith("linux"):
        return ".so"
    raise NativeBuildError(f"Native wheels are not supported on platform {name!r}")


def library_filename(library: str, platform_name: str | None = None) -> str:
    name = sys.platform if platform_name is None else platform_name
    # 'rolling.so' would shadow rolling.py as a CPython extension module. These
    # libraries expose a plain C ABI and must only be loaded through ctypes.
    prefix = "lib" if name.startswith("linux") else ""
    return prefix + library + library_suffix(name)


def macos_architectures() -> tuple[str, ...]:
    """Architectures compiled into the dylibs, independent of Python's slices."""
    arguments = shlex.split(os.environ.get("ARCHFLAGS", ""))
    architectures = []
    for position, argument in enumerate(arguments):
        if argument == "-arch":
            if position + 1 == len(arguments):
                raise NativeBuildError("ARCHFLAGS ends with -arch without an architecture")
            architectures.append(arguments[position + 1])
    if not architectures:
        # platform.machine() follows the running process on macOS, including an
        # x86-64 Python translated by Rosetta.  A universal2 interpreter's build
        # configuration does not describe the libraries compiled by this run.
        if struct.calcsize("P") != 8:
            raise NativeBuildError("macOS native builds require 64-bit Python")
        machine = platform.machine().lower()
        architectures = [{"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)]
    unique = tuple(sorted(set(architectures)))
    if not set(unique).issubset({"x86_64", "arm64"}):
        raise NativeBuildError(f"Unsupported macOS native architectures: {unique!r}")
    return unique


def native_wheel_platform_tag(platform_tag: str, platform_name: str | None = None) -> str:
    """Correct macOS Python-distribution tags to match our compiled libraries."""
    name = sys.platform if platform_name is None else platform_name
    if name != "darwin":
        return platform_tag
    fields = platform_tag.split("_", 3)
    if len(fields) != 4 or fields[0] != "macosx":
        raise NativeBuildError(f"Invalid macOS wheel platform tag: {platform_tag!r}")
    architectures = macos_architectures()
    architecture_tag = "universal2" if len(architectures) == 2 else architectures[0]
    return "_".join([*fields[:3], architecture_tag])


def _compiler_command(cc: str | None, platform_name: str) -> list[str]:
    requested = cc or os.environ.get("QLIB_CC") or os.environ.get("CC")
    requested = requested or ("clang" if platform_name == "darwin" else "gcc")
    # An executable path containing spaces is a single argument.  CC may also
    # contain a compiler wrapper and flags, as in 'ccache gcc'.
    resolved = shutil.which(requested)
    if resolved:
        return [resolved]
    command = shlex.split(requested, posix=platform_name != "win32")
    command = [part.strip('"') if platform_name == "win32" else part for part in command]
    if command and shutil.which(command[0]):
        command[0] = shutil.which(command[0]) or command[0]
        return command
    requirement = "MinGW-w64 GCC" if platform_name == "win32" else "a C11 compiler"
    raise NativeBuildError(
        f"{requirement} was not found: {requested!r}. "
        "Install the compiler or select it with QLIB_CC / --cc. "
        "A native wheel cannot be built without compiling its C libraries."
    )


def _compiler_identity(command: list[str], platform_name: str) -> tuple[str, str]:
    try:
        target = subprocess.check_output([*command, "-dumpmachine"], text=True).strip()
        version = subprocess.check_output([*command, "--version"], text=True).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError) as exc:
        raise NativeBuildError(f"Cannot inspect C compiler {command[0]!r}: {exc}") from exc
    if platform_name == "win32":
        if platform.machine().lower() in ("arm64", "aarch64"):
            raise NativeBuildError("Windows native builds currently require x86 or x86-64 Python")
        architecture = "x86_64" if struct.calcsize("P") == 8 else "i686"
        if "mingw32" not in target or "gcc" not in version.lower():
            raise NativeBuildError(f"MinGW-w64 GCC is required; got {version} ({target})")
        if not target.startswith(architecture + "-"):
            raise NativeBuildError(
                f"Compiler target {target} does not match this Python ({architecture})"
            )
    elif platform_name == "darwin":
        if "clang" not in version.lower() or "darwin" not in target:
            raise NativeBuildError(f"macOS Clang is required; got {version} ({target})")
    elif not platform_name.startswith("linux"):
        library_suffix(platform_name)
    elif "linux" not in target:
        raise NativeBuildError(f"A Linux compiler is required; got {version} ({target})")
    else:
        machine = platform.machine().lower()
        aliases = {"amd64": "x86_64", "arm64": "aarch64"}
        architecture = aliases.get(machine, machine)
        if struct.calcsize("P") == 4 and architecture == "x86_64":
            architecture = "i686"
        target_architecture = target.split("-", 1)[0]
        if target_architecture != architecture:
            raise NativeBuildError(
                f"Compiler target {target} does not match this Python ({architecture})"
            )
    return target, version


def compile_command(
    compiler: list[str], source: Path, output: Path, platform_name: str,
) -> list[str]:
    flags = list(COMMON_FLAGS)
    if platform_name == "win32":
        flags += ["-shared", "-static-libgcc"]
    elif platform_name == "darwin":
        flags += ["-dynamiclib", "-Wl,-install_name,@rpath/" + output.name]
        architecture_flags = shlex.split(os.environ.get("ARCHFLAGS", ""))
        architectures = macos_architectures()
        flags += architecture_flags
        if "-arch" not in architecture_flags:
            flags += ["-arch", architectures[0]]
        deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        if deployment_target:
            flags.append("-mmacosx-version-min=" + deployment_target)
    elif platform_name.startswith("linux"):
        flags += ["-fPIC", "-shared", "-Wl,--no-undefined"]
    else:
        library_suffix(platform_name)
    return [*compiler, *flags, str(source), "-o", str(output), "-lm"]


def build_libraries(
    source_directory: Path | str,
    output_directory: Path | str,
    *,
    cc: str | None = None,
    names: tuple[str, ...] = LIBRARIES,
    platform_name: str | None = None,
) -> list[Path]:
    """Compile fresh libraries; never reuse binaries from a source checkout."""
    name = sys.platform if platform_name is None else platform_name
    library_suffix(name)
    if not names or any(item not in LIBRARIES for item in names):
        raise NativeBuildError(f"Unknown native library selection: {names!r}")
    source_directory = Path(source_directory).resolve()
    output_directory = Path(output_directory).resolve()
    compiler = _compiler_command(cc, name)
    target, version = _compiler_identity(compiler, name)
    print(f"Building Qlib native libraries with {version} ({target})", flush=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    # Only publish complete builds. The temporary directory is on the destination
    # volume, so a failed compiler invocation cannot leave a partial library.
    with tempfile.TemporaryDirectory(prefix=".qlib-native-", dir=output_directory) as staging:
        staging_directory = Path(staging)
        for item in names:
            source = source_directory / (item + ".c")
            if not source.is_file():
                raise NativeBuildError(f"Missing C source: {source}")
            temporary_output = staging_directory / library_filename(item, name)
            command = compile_command(compiler, source, temporary_output, name)
            try:
                subprocess.run(command, check=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                raise NativeBuildError(f"Failed to build {item} with {compiler[0]}: {exc}") from exc
            if not temporary_output.is_file():
                raise NativeBuildError(f"Compiler did not produce {temporary_output.name}")
        for item in names:
            destination = output_directory / library_filename(item, name)
            (staging_directory / destination.name).replace(destination)
            outputs.append(destination)
            print(f"Built {destination}", flush=True)
    return outputs
