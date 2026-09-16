# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Negative controls for the integration CI's required device evidence."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/kernel_ci_results.py"
_SPEC = importlib.util.spec_from_file_location("kernel_ci_results", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _xml(path, cases, **counts):
    """Write a minimal pytest-style JUnit report."""
    attributes = {"tests": len(cases), "errors": 0, "failures": 0, "skipped": 0} | counts
    attrs = " ".join(f'{name}="{value}"' for name, value in attributes.items())
    path.write_text(f"<testsuites><testsuite {attrs}>{''.join(cases)}</testsuite></testsuites>")
    return path


def test_passed_device_report(tmp_path):
    path = _xml(tmp_path / "device.xml", ['<testcase name="scalar"/>', '<testcase name="capture"/>'])
    message, ok = _MODULE.report(path)
    assert ok
    assert "2 passed" in message


@pytest.mark.parametrize("content", [None, "", "broken", "<unexpected/>", "<testsuites/>"])
def test_missing_or_invalid_report_fails(tmp_path, content):
    path = tmp_path / "missing.xml"
    if content is not None:
        path.write_text(content)
    message, ok = _MODULE.report(path)
    assert not ok
    assert "MISSING/INVALID" in message


@pytest.mark.parametrize("outcome", ["skipped", "failure", "error"])
def test_nonpassing_device_case_fails(tmp_path, outcome):
    path = _xml(tmp_path / "device.xml", ['<testcase name="good"/>', f"<testcase><{outcome}/></testcase>"])
    assert not _MODULE.report(path)[1]


@pytest.mark.parametrize(
    "counts", [{"tests": 2}, {"failures": 1}, {"errors": 1}, {"skipped": 1}, {"tests": "invalid"}]
)
def test_inconsistent_or_failed_suite_fails(tmp_path, counts):
    path = _xml(tmp_path / "device.xml", ["<testcase/>"], **counts)
    assert not _MODULE.report(path)[1]


def test_cpu_skip_policy_cannot_accept_empty_or_failed_suite(tmp_path):
    path = _xml(tmp_path / "unit.xml", ["<testcase><skipped/></testcase>"], skipped=1)
    assert not _MODULE.report(path, allow_skips=True)[1]
    _xml(path, ["<testcase/>", "<testcase><skipped/></testcase>"], skipped=1)
    assert _MODULE.report(path, allow_skips=True)[1]
    assert not _MODULE.report(path)[1]
    _xml(path, ["<testcase/>", "<testcase><failure/></testcase>"], failures=1)
    assert not _MODULE.report(path, allow_skips=True)[1]


def test_cli_requires_every_named_report(tmp_path):
    eager = _xml(tmp_path / "eager.xml", ["<testcase/>"])
    capture = tmp_path / "capture.xml"
    command = [sys.executable, str(_SCRIPT), str(eager), str(capture)]
    missing = subprocess.run(command, capture_output=True, text=True, check=False)
    assert missing.returncode == 1
    assert "eager.xml | PASS" in missing.stdout
    assert "capture.xml | MISSING/INVALID" in missing.stdout
    _xml(capture, ["<testcase/>"])
    assert subprocess.run(command, capture_output=True, check=False).returncode == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
