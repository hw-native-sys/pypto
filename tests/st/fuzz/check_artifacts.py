# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate artifacts produced by --save-kernels and --dump-passes.

Checks that each test case output directory contains the expected files:
  kernels/*.cpp, orchestration/orchestrator.cpp, kernel_config.py,
  golden.py, passes_dump/*.py, and report/*.
"""

import argparse
import sys
from pathlib import Path


def _has_cpp_files(kernels_dir: Path) -> bool:
    """Check that kernels/ has at least one .cpp in aic/ or aiv/."""
    for sub in ("aic", "aiv"):
        sub_dir = kernels_dir / sub
        if sub_dir.is_dir() and list(sub_dir.glob("*.cpp")):
            return True
    return False


def _check_test_dir(test_dir: Path) -> list[str]:
    """Return list of missing artifact descriptions for one test directory."""
    missing: list[str] = []

    # kernels/*.cpp
    kernels_dir = test_dir / "kernels"
    if not kernels_dir.is_dir() or not _has_cpp_files(kernels_dir):
        missing.append("kernels/*.cpp (no .cpp in aic/ or aiv/)")

    # orchestration/orchestrator.cpp
    orch_file = test_dir / "orchestration" / "orchestrator.cpp"
    if not orch_file.is_file() or orch_file.stat().st_size == 0:
        missing.append("orchestration/orchestrator.cpp")

    # kernel_config.py
    kc = test_dir / "kernel_config.py"
    if not kc.is_file() or kc.stat().st_size == 0:
        missing.append("kernel_config.py")

    # golden.py
    gp = test_dir / "golden.py"
    if not gp.is_file() or gp.stat().st_size == 0:
        missing.append("golden.py")

    # passes_dump/*.py
    pd = test_dir / "passes_dump"
    if not pd.is_dir() or not list(pd.glob("*.py")):
        missing.append("passes_dump/*.py (no .py files)")

    # report/
    rp = test_dir / "report"
    if not rp.is_dir() or not list(rp.iterdir()):
        missing.append("report/ (empty or missing)")

    return missing


def check_artifacts(artifacts_dir: str) -> bool:
    """Validate all test directories under artifacts_dir.

    Args:
        artifacts_dir: Path to the directory containing numbered test subdirs.

    Returns:
        True if all test directories have complete artifacts.
    """
    root = Path(artifacts_dir)
    if not root.is_dir():
        print(f"FAILED: artifacts directory not found: {artifacts_dir}")
        return False

    # Discover test subdirectories (sorted for stable output)
    test_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name[0].isdigit())

    if not test_dirs:
        print(f"FAILED: no test directories found in {artifacts_dir}")
        return False

    print("=" * 60)
    print(f"Artifact Check ({len(test_dirs)} test directories)")
    print("=" * 60)
    print()

    failures: list[tuple[str, list[str]]] = []

    for td in test_dirs:
        missing = _check_test_dir(td)
        if missing:
            failures.append((td.name, missing))
            print(f"  FAIL  {td.name}")
            for m in missing:
                print(f"        - {m}")
        else:
            print(f"  OK    {td.name}")

    print()
    complete = len(test_dirs) - len(failures)
    print(f"Artifacts: {complete}/{len(test_dirs)} complete")

    if failures:
        print(f"FAILED: {len(failures)} test(s) missing artifacts")
        return False

    print("PASSED: all test directories have complete artifacts")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate --save-kernels and --dump-passes artifacts.")
    parser.add_argument(
        "artifacts_dir",
        help="Path to the directory containing test output subdirectories",
    )
    args = parser.parse_args()

    if not check_artifacts(args.artifacts_dir):
        sys.exit(1)


if __name__ == "__main__":
    main()
