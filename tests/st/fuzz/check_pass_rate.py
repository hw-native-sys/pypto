# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check fuzz test pass rate from JUnit XML results.

Supports one or more JUnit XML files. When multiple files are given (e.g. sim
and hardware), a test case counts as passing only if it passes in ALL files.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

_TRACEBACK_TAIL_LINES = 30


def _parse_suites(xml_path: str) -> list[ET.Element]:
    """Parse JUnit XML and return list of <testsuite> elements."""
    root = ET.parse(xml_path).getroot()
    return root.findall("testsuite") if root.tag == "testsuites" else [root]


def _get_test_results(suites: list[ET.Element]) -> dict[str, bool]:
    """Return mapping of test name -> passed for all non-skipped tests."""
    results: dict[str, bool] = {}
    for suite in suites:
        for tc in suite.findall("testcase"):
            if tc.find("skipped") is not None:
                continue
            name = tc.attrib.get("name", "unknown")
            results[name] = tc.find("failure") is None and tc.find("error") is None
    return results


def _print_failures(suites: list[ET.Element]) -> None:
    """Print details for each failed or errored test case.

    Args:
        suites: List of <testsuite> elements from the JUnit XML.
    """
    failed_cases: list[tuple[str, str, str]] = []
    for suite in suites:
        for tc in suite.findall("testcase"):
            node = tc.find("failure") or tc.find("error")
            if node is None:
                continue
            name = tc.attrib.get("name", "unknown")
            msg = node.attrib.get("message", "")
            text = (node.text or "").strip()
            failed_cases.append((name, msg, text))

    if not failed_cases:
        return

    sep = "-" * 60
    print(f"Failed test details ({len(failed_cases)}):")
    for name, msg, text in failed_cases:
        print(sep)
        print(f"FAILED: {name}")
        if msg:
            print(msg)
        if text:
            lines = text.splitlines()
            if len(lines) > _TRACEBACK_TAIL_LINES:
                print(f"  ... ({len(lines) - _TRACEBACK_TAIL_LINES} lines omitted) ...")
                lines = lines[-_TRACEBACK_TAIL_LINES:]
            print("\n".join(lines))
    print(sep)
    print()


def _print_single_stats(xml_path: str, suites: list[ET.Element], threshold: float) -> tuple[int, int]:
    """Print stats for one XML file. Returns (passed, executed)."""
    total = sum(int(s.attrib.get("tests", 0)) for s in suites)
    failures = sum(int(s.attrib.get("failures", 0)) for s in suites)
    errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
    skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)

    executed = total - skipped
    passed = executed - failures - errors
    pass_rate = passed / executed if executed > 0 else 0.0

    print(f"Results [{xml_path}]:")
    print(f"  Total: {total}  Passed: {passed}  Failed: {failures}  Errors: {errors}  Skipped: {skipped}")
    print(f"  Pass rate: {pass_rate:.1%}")
    print()

    if failures + errors > 0:
        _print_failures(suites)

    return passed, executed


def check_pass_rate(xml_paths: list[str], threshold: float) -> bool:
    """Parse one or more JUnit XMLs and check if combined pass rate meets threshold.

    Single XML: standard per-test pass rate.
    Multiple XMLs: pass rates are pooled across all platforms — total passed
    divided by total executions (sim + hardware combined).

    Missing files are skipped with a warning; if all files are missing the
    check fails.

    Args:
        xml_paths: Paths to JUnit XML results files.
        threshold: Minimum pass rate (0.0-1.0).

    Returns:
        True if pass rate meets threshold.
    """
    # Filter to existing files, warn about missing ones
    available: list[str] = []
    for path in xml_paths:
        if os.path.exists(path):
            available.append(path)
        else:
            print(f"WARNING: result file not found, skipping: {path}")

    if not available:
        print("FAILED: no result files found")
        return False

    # --- Single XML: original behaviour ---
    if len(available) == 1:
        suites = _parse_suites(available[0])
        passed, executed = _print_single_stats(available[0], suites, threshold)

        if executed == 0:
            print("FAILED: no tests were executed")
            return False

        pass_rate = passed / executed
        if pass_rate >= threshold:
            print(f"PASSED: {pass_rate:.1%} >= {threshold:.1%}")
            return True
        print(f"FAILED: {pass_rate:.1%} < {threshold:.1%}")
        return False

    # --- Multiple XMLs: pooled pass rate ---
    print("=" * 60)
    print(f"Fuzz Test Results (pooled, {len(available)} platforms)")
    print("=" * 60)
    print()

    total_passed = 0
    total_executed = 0
    for path in available:
        suites = _parse_suites(path)
        passed, executed = _print_single_stats(path, suites, threshold)
        total_passed += passed
        total_executed += executed

    if total_executed == 0:
        print("FAILED: no tests were executed across any platform")
        return False

    combined_rate = total_passed / total_executed

    print("=" * 60)
    print("Combined (pooled across all platforms):")
    print(f"  Executions: {total_executed}")
    print(f"  Passed:     {total_passed}")
    print(f"  Failed:     {total_executed - total_passed}")
    print(f"  Pass rate:  {combined_rate:.1%} (threshold: {threshold:.1%})")
    print("=" * 60)
    print()

    if combined_rate >= threshold:
        print(f"PASSED: {combined_rate:.1%} >= {threshold:.1%}")
        return True
    print(f"FAILED: {combined_rate:.1%} < {threshold:.1%}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check fuzz test pass rate from JUnit XML. "
        "Pass multiple XML files to compute a combined (sim+hw) pass rate."
    )
    parser.add_argument(
        "xml_paths",
        nargs="+",
        help="Path(s) to JUnit XML results file(s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum pass rate, 0.0-1.0 (default: 0.8)",
    )
    args = parser.parse_args()

    if args.threshold < 0.0 or args.threshold > 1.0:
        print(f"Error: threshold must be between 0.0 and 1.0, got {args.threshold}")
        sys.exit(2)

    if not check_pass_rate(args.xml_paths, args.threshold):
        sys.exit(1)


if __name__ == "__main__":
    main()
