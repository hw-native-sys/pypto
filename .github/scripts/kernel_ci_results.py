# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Require usable JUnit evidence; skipped device tests never count as acceptance."""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def report(path: Path, *, allow_skips: bool = False) -> tuple[str, bool]:
    """Summarize one report, failing missing, empty, inconsistent or failed evidence."""
    try:
        root = ET.parse(path).getroot()
        if root.tag not in ("testsuites", "testsuite"):
            raise ValueError(f"unexpected root {root.tag!r}")
        cases = list(root.iter("testcase"))
        if not cases:
            raise ValueError("no test cases recorded")
        suites = list(root.iter("testsuite"))
        if not suites or sum(int(s.attrib["tests"]) for s in suites) != len(cases):
            raise ValueError("test count does not match recorded cases")
        failed = sum(case.find("failure") is not None or case.find("error") is not None for case in cases)
        skipped = sum(case.find("skipped") is not None for case in cases)
        # Suite-level setup/collection failures must not disappear just because
        # a producer failed to attach them to a testcase.
        errors = sum(int(s.get("errors", "0")) + int(s.get("failures", "0")) for s in suites)
        if sum(int(s.get("skipped", "0")) for s in suites) != skipped:
            raise ValueError("skip count does not match recorded cases")
        passed = len(cases) - failed - skipped
        ok = passed > 0 and failed == 0 and errors == 0 and (allow_skips or skipped == 0)
        status = "PASS" if ok else "FAIL"
        return (
            f"{status}: {passed} passed, {failed} failed, {skipped} skipped, {errors} suite errors/failures",
            ok,
        )
    except (OSError, ET.ParseError, KeyError, ValueError) as exc:
        return f"MISSING/INVALID: {exc}", False


def main() -> int:
    """Print a Markdown table and fail unless every requested report is acceptable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-skips", action="store_true", help="For CPU UT only, not device acceptance")
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    print("| Report | Result |\n| --- | --- |")
    accepted = True
    for path in args.reports:
        message, ok = report(path, allow_skips=args.allow_skips)
        # Report names/messages are data, never Markdown structure.
        label = path.name.replace("|", "\\|").replace("\n", " ")
        message = message.replace("|", "\\|").replace("\n", " ")
        print(f"| {label} | {message} |")
        accepted = accepted and ok
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
