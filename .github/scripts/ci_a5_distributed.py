# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pin an integration PR merge and report its independent A5 distributed check."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ci_control import api, output, report, sha

BRANCH = "integration/urma-ccu-test"
COMMAND = "/ci a5-distributed"
JOB = "onboard-tests-a5-distributed"


def prepare(repo: str, event: dict[str, Any]) -> None:
    """Accept an exact new PR command and freeze its current head/base/merge."""
    if (
        event.get("action") != "created"
        or "pull_request" not in event.get("issue", {})
        or event.get("comment", {}).get("body", "").strip() != COMMAND
    ):
        return
    number = int(event["issue"]["number"])
    deadline = time.monotonic() + 60
    pinned = None
    while True:
        pr = api(f"repos/{repo}/pulls/{number}")
        if (
            pr["state"] != "open"
            or pr["base"]["ref"] != BRANCH
            or pr["base"]["repo"]["full_name"] != repo
            or not pr["head"]["repo"]
        ):
            raise ValueError(f"{COMMAND} requires an open PR targeting {repo}:{BRANCH}")
        ref = api(f"repos/{repo}/git/ref/heads/{BRANCH}")
        if ref.get("ref") != f"refs/heads/{BRANCH}" or ref.get("object", {}).get("type") != "commit":
            raise ValueError("Expected the integration branch to resolve to a commit")
        head, base = sha(pr["head"]["sha"]), sha(ref["object"]["sha"])
        if pinned is None:
            pinned = (head, base)
        elif pinned != (head, base):
            raise ValueError(f"PR head or base changed while waiting; post {COMMAND} again")
        if pr.get("mergeable") is False:
            raise ValueError("PR has merge conflicts; update the PR branch before retrying")
        if pr.get("mergeable") is True and pr.get("merge_commit_sha"):
            merge = sha(pr["merge_commit_sha"])
            commit = api(f"repos/{repo}/git/commits/{merge}")
            if [parent["sha"] for parent in commit["parents"]] == [base, head]:
                break
        # GitHub can briefly return an old synthetic merge after a base update.
        # Retry only that eventual consistency; never switch to a newer head.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f"GitHub test merge is not ready; retry {COMMAND} later")
        time.sleep(min(5, remaining))
    url = (
        f"https://github.com/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        f"/attempts/{os.environ['GITHUB_RUN_ATTEMPT']}"
    )
    # Each request owns its own check ID, so a late result cannot overwrite a
    # newer request. No ci/full status is written by this partial test run.
    check = api(
        f"repos/{repo}/check-runs",
        {"name": "ci/a5-distributed", "head_sha": head, "status": "in_progress", "details_url": url},
    )
    if type(check.get("id")) is not int or check["id"] <= 0:
        raise ValueError("GitHub did not return a valid check ID")
    output({"checkout_sha": merge, "check_id": str(check["id"])})
    report(f"A5 distributed tests for PR #{number}: head {head}, base {base}, merge {merge}. [Run]({url})")


def finish(repo: str, needs: dict[str, Any]) -> None:
    """Only a completed device job can produce a successful partial-test check."""
    check_id = needs["changes"]["outputs"]["check_id"]
    if not check_id.isdecimal() or int(check_id) <= 0:
        raise ValueError("Expected a positive check ID")
    expected = {"changes", "toolchain", JOB}
    if set(needs) != expected:
        raise ValueError("Unexpected A5 distributed dependency set")
    failed = [name for name in sorted(expected) if needs[name]["result"] != "success"]
    summary = ", ".join(f"{name}: {needs[name]['result']}" for name in sorted(expected))
    api(
        f"repos/{repo}/check-runs/{check_id}",
        {
            "status": "completed",
            "conclusion": "failure" if failed else "success",
            "output": {"title": "A5 distributed test results", "summary": summary},
        },
        method="PATCH",
    )
    if failed:
        raise ValueError(f"A5 distributed tests did not pass: {summary}")


def main() -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    if sys.argv[1] == "prepare":
        prepare(repo, json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()))
    elif sys.argv[1] == "finish":
        finish(repo, json.loads(os.environ["CI_NEEDS"]))
    else:
        raise ValueError("Expected prepare or finish")


if __name__ == "__main__":
    main()
