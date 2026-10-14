# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Restore the verbatim issue #2979 baseline, checking its recorded SHA-256 hashes."""

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

BASELINE = Path(__file__).resolve().parent / "baseline"
ISSUE_API = "https://api.github.com/repos/hw-native-sys/pypto/issues/2979"


def restore_baseline(issue_body: str) -> None:
    """Validate all four source blocks before writing any baseline file."""
    manifest = json.loads((BASELINE / "sources.json").read_text())
    sources: dict[str, bytes] = {}
    for name, expected_sha in manifest["sha256"].items():
        pattern = rf"<summary>{re.escape(name)}[^<]*</summary>\s*```[^\n]*\n(.*?)\n```"
        matches = re.findall(pattern, issue_body, flags=re.DOTALL)
        if len(matches) != 1:
            raise ValueError(f"Expected one source block for {name}, got {len(matches)}")
        source = (matches[0] + "\n").encode()
        actual_sha = hashlib.sha256(source).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"Issue source changed for {name}: expected {expected_sha}, got {actual_sha}")
        destination = BASELINE / name
        if destination.exists() and destination.read_bytes() != source:
            raise ValueError(f"Refusing to overwrite modified baseline file: {destination}")
        sources[name] = source
    for name, source in sources.items():
        (BASELINE / name).write_bytes(source)
        print(f"Verified {name}: {manifest['sha256'][name]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-json", type=Path, help="Use a saved GitHub issue API response offline")
    args = parser.parse_args()
    if args.issue_json is not None:
        issue = json.loads(args.issue_json.read_text())
    else:
        request = urllib.request.Request(ISSUE_API, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            issue = json.load(response)
    restore_baseline(issue["body"])


if __name__ == "__main__":
    main()
