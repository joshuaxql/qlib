"""Build the pure C rolling/expanding/PIT DLLs using MinGW-w64 GCC, never MSVC.

Usage: python scripts/build_rolling.py --cc D:/software/mingw64/bin/gcc.exe
"""

import argparse
from pathlib import Path
import shutil
import struct
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="gcc", help="MinGW-w64 GCC executable (default: gcc on PATH)")
    parser.add_argument("--only", choices=("rolling", "expanding", "pit"), help="Build just one DLL (default: all)")
    args = parser.parse_args()
    compiler = shutil.which(args.cc)
    if compiler is None:
        parser.error(f"MinGW-w64 GCC not found: {args.cc}")
    target = subprocess.check_output([compiler, "-dumpmachine"], text=True).strip()
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    if "mingw32" not in target or "gcc" not in version.lower():
        parser.error(f"MinGW-w64 GCC is required; got {version} ({target})")
    architecture = "x86_64" if struct.calcsize("P") == 8 else "i686"
    if not target.startswith(architecture + "-"):
        parser.error(f"Compiler target {target} does not match this Python ({architecture})")
    if sys.platform != "win32":
        parser.error("Run this MinGW DLL build with Windows Python")
    directory = Path(__file__).resolve().parents[1] / "qlib" / "data" / "_libs"
    print(f"Compiler: {version}\nTarget: {target}", flush=True)
    for name in ((args.only,) if args.only else ("rolling", "expanding", "pit")):
        output = directory / f"{name}.dll"
        command = [
            compiler, "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-fno-fast-math", "-ffp-contract=off", "-shared", "-static-libgcc",
            str(directory / f"{name}.c"), "-o", str(output), "-lm",
        ]
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True)
        print(f"Built: {output}")


if __name__ == "__main__":
    main()
