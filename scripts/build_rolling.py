"""Build the pure C rolling/expanding/PIT libraries for the current platform.

Usage: python scripts/build_rolling.py --cc D:/software/mingw64/bin/gcc.exe
Windows requires MinGW-w64 GCC; Linux uses GCC and macOS uses Clang.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_native import NativeBuildError, build_libraries  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", help="Compiler executable (default: QLIB_CC / CC / platform compiler)")
    parser.add_argument("--only", choices=("rolling", "expanding", "pit"), help="Build one library (default: all)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: qlib/data/_libs)")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "qlib" / "data" / "_libs"
    options = {"names": (args.only,)} if args.only else {}
    try:
        build_libraries(directory, args.output_dir or directory, cc=args.cc, **options)
    except NativeBuildError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
