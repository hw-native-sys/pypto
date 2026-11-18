# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pin a main PR merge and report its independent A5 distributed check."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ci_control import (
    MERGE_POLL_SECONDS,
    MERGE_WAIT_SECONDS,
    api,
    current_base,
    matching_lint_run,
    output,
    pages,
    pull,
    report,
    sha,
)

COMMAND = "/ci a5-distributed"
JOB = "onboard-tests-a5-distributed"


def attempt_url(repo: str, run_id: str, attempt: str) -> str:
    return f"https://github.com/{repo}/actions/runs/{run_id}/attempts/{attempt}"


def attempt_fields(url: str, summary: str) -> dict[str, Any]:
    # GitHub Actions may replace details_url with its native check page. The
    # first summary line remains both clickable and an exact attempt marker.
    return {
        "details_url": url,
        "output": {"title": "A5 distributed tests", "summary": f"[Workflow attempt]({url})\n\n{summary}"},
    }


def update_check(repo: str, check_id: str, payload: dict[str, Any]) -> None:
    """Retry an idempotent PATCH on transient service/transport errors only."""
    for retry in range(3):
        try:
            api(f"repos/{repo}/check-runs/{check_id}", payload, method="PATCH")
            return
        except subprocess.CalledProcessError as error:
            transient = re.search(
                r"HTTP (?:429|5[0-9]{2})\b|connection reset|timeout|unexpected EOF", error.stderr or "", re.I
            )
            if not transient or retry == 2:
                raise
            time.sleep(2**retry)


def head_checks(repo: str, head: str) -> list[dict[str, Any]]:
    path = f"repos/{repo}/commits/{head}/check-runs?check_name=ci%2Fa5-distributed&filter=all"
    return sorted(pages(path, "check_runs"), key=lambda check: check["id"], reverse=True)


def reusable_request(repo: str, number: int, checks: list[dict[str, Any]], run_id: str) -> str | None:
    """Look up the owning workflow so cancelled or failed publishers never block a retry."""
    owners: dict[str, list[dict[str, Any]]] = {}
    for check in checks:
        owner = re.fullmatch(r"([1-9][0-9]*)(?::[1-9][0-9]*:[1-9][0-9]*)?", check["external_id"] or "")
        if owner and owner[1] != run_id and check["app"]["slug"] == "github-actions":
            owners.setdefault(owner[1], []).append(check)
    for owner_id, owned in owners.items():
        try:
            run = api(f"repos/{repo}/actions/runs/{owner_id}")
        except subprocess.CalledProcessError as error:
            if "HTTP 404" in (error.stderr or ""):
                continue  # The historical workflow run has been deleted.
            raise
        if (
            run["path"] != ".github/workflows/ci-a5-distributed.yml"
            or run["event"] != "issue_comment"
            or run["head_branch"] != "main"
            or not re.fullmatch(rf"A5 distributed PR #{number} \(comment [1-9][0-9]*\)", run["display_title"])
        ):
            continue
        url = attempt_url(repo, owner_id, str(run["run_attempt"]))
        if run["status"] != "completed" or (
            run["conclusion"] == "success"
            and any(check["conclusion"] == "success" and same_attempt(check, url) for check in owned)
        ):
            return url
    return None


def same_attempt(check: dict[str, Any], url: str) -> bool:
    return (check["output"]["summary"] or "").startswith(f"[Workflow attempt]({url})\n")


def attempt_check(repo: str, original_id: str, run_id: str, attempt: str) -> str:
    """Select or create the check for a rerun under the shared writer queue."""
    if attempt == "1":
        return original_id
    original = api(f"repos/{repo}/check-runs/{original_id}")
    if original["name"] != "ci/a5-distributed" or original["external_id"] != run_id:
        raise ValueError("Unexpected A5 check ownership")
    url = attempt_url(repo, run_id, attempt)
    head = sha(original["head_sha"])
    key = f"{run_id}:{attempt}:{original_id}"
    family = [
        check
        for check in head_checks(repo, head)
        if check["app"]["id"] == original["app"]["id"]
        and re.fullmatch(rf"{run_id}:[1-9][0-9]*:{original_id}", check["external_id"] or "")
    ]
    family.append(original)
    current = next((c for c in family if c["external_id"] == key or same_attempt(c, url)), None)
    pending = [c for c in family if c["status"] != "completed"]
    if current or pending:
        selected = current or pending[0]
        for check in pending:
            if check["id"] != selected["id"]:
                # Earlier versions could strand multiple checks after repeated
                # publisher outages. Retire only siblings of this exact request.
                update_check(repo, str(check["id"]), {"status": "completed", "conclusion": "cancelled"})
        if current is None:
            update_check(repo, str(selected["id"]), attempt_fields(url, "Retrying the distributed check."))
        return str(selected["id"])
    # GitHub retains the conclusion of a completed check after an in_progress
    # PATCH. Only a rerun with no unfinished check needs a fresh one.
    created = api(
        f"repos/{repo}/check-runs",
        {
            "name": original["name"],
            "head_sha": head,
            "external_id": key,
            "status": "in_progress",
            **attempt_fields(url, "Rerunning the distributed check."),
        },
    )
    if type(created.get("id")) is not int or created["id"] <= 0:
        raise ValueError("GitHub did not return a valid rerun check ID")
    return str(created["id"])


def reopen(repo: str, run: dict[str, Any]) -> None:
    """Refresh a carried-over PR check when GitHub reruns the device job alone."""
    if (
        run["path"] != ".github/workflows/ci-a5-distributed.yml"
        or run["event"] != "issue_comment"
        or run["head_branch"] != "main"
        or run["run_attempt"] < 2
    ):
        return
    run_id, attempt = str(run["id"]), str(run["run_attempt"])
    # The source workflow puts the trusted prepare output in the device job
    # name. Never derive a check ID from a PR comment or a test artifact.
    for _ in range(12):
        live = api(f"repos/{repo}/actions/runs/{run_id}")
        if live["status"] == "completed" or live["run_attempt"] != run["run_attempt"]:
            return
        jobs = pages(f"repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs", "jobs")
        for job in jobs:
            match = re.fullmatch(r"onboard-tests-a5-distributed / check ([1-9][0-9]*)", job["name"])
            if not match or job["run_attempt"] != run["run_attempt"]:
                continue
            attempt_check(repo, match[1], run_id, attempt)
            return
        if all(job["status"] == "completed" for job in jobs) and any(job["name"] == "result" for job in jobs):
            return
        time.sleep(5)
    raise ValueError("Could not identify the A5 rerun job")


def pinned_merge(repo: str, number: int) -> tuple[dict[str, Any], str, str, str]:
    """Wait for GitHub to publish the test merge for the first head/base observed."""
    deadline = time.monotonic() + MERGE_WAIT_SECONDS
    pinned = None
    while True:
        pr = pull(repo, number)
        head, base = sha(pr["head"]["sha"]), current_base(repo)
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
                return pr, head, base, merge
        # GitHub can briefly return an old synthetic merge after a base update.
        # Retry only that eventual consistency; never switch to a newer head.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f"GitHub test merge is not ready; retry {COMMAND} later")
        time.sleep(min(MERGE_POLL_SECONDS, remaining))


def prepare(repo: str, event: dict[str, Any]) -> None:
    """Accept an exact new PR command and freeze its current head/base/merge."""
    if (
        event.get("action") != "created"
        or "pull_request" not in event.get("issue", {})
        or event.get("comment", {}).get("body", "").strip() != COMMAND
    ):
        return
    number = int(event["issue"]["number"])
    pr, head, base, merge = pinned_merge(repo, number)
    lint = matching_lint_run(repo, pr, COMMAND)
    if lint["conclusion"] != "success":
        raise ValueError(f"PR lint workflow did not pass; fix or rerun lint before {COMMAND}")
    run_id = os.environ["GITHUB_RUN_ID"]
    checks = head_checks(repo, head)
    existing = reusable_request(repo, number, checks, run_id)
    if existing:
        report(f"Reusing the A5 distributed run for PR #{number} at {head}. [Run]({existing})")
        return
    latest = pull(repo, number)
    if latest["head"]["sha"] != head or latest.get("merge_commit_sha") != merge or current_base(repo) != base:
        raise ValueError(f"PR head or base changed during validation; post {COMMAND} again")
    url = attempt_url(repo, run_id, os.environ["GITHUB_RUN_ATTEMPT"])
    for previous in checks:
        if previous["external_id"] == run_id and previous["app"]["slug"] == "github-actions":
            # Re-running all jobs also retries prepare. Pass its original ID to
            # the serialized observer and publisher, just like a job-only rerun.
            output({"checkout_sha": merge, "check_id": str(previous["id"])})
            report(f"Retrying A5 distributed tests for PR #{number} at {head}. [Run]({url})")
            return
    # Each request owns its own check ID, so a late result cannot overwrite a
    # newer request. No ci/full status is written by this partial test run.
    check = api(
        f"repos/{repo}/check-runs",
        {
            "name": "ci/a5-distributed",
            "head_sha": head,
            "external_id": run_id,
            "status": "in_progress",
            **attempt_fields(url, "Preparing the distributed check."),
        },
    )
    if type(check.get("id")) is not int or check["id"] <= 0:
        raise ValueError("GitHub did not return a valid check ID")
    check_id = str(check["id"])
    try:
        output({"checkout_sha": merge, "check_id": check_id})
    except (OSError, ValueError):
        # Without these outputs the normal finalizer cannot find the check.
        try:
            update_check(
                repo,
                check_id,
                {
                    "status": "completed",
                    "conclusion": "failure",
                    **attempt_fields(url, "Could not publish job outputs."),
                },
            )
        except subprocess.CalledProcessError as cleanup_error:
            report(f"Could not complete check {check_id}: {cleanup_error}")
        raise
    report(f"A5 distributed tests for PR #{number}: head {head}, base {base}, merge {merge}. [Run]({url})")


def finish(repo: str, needs: dict[str, Any]) -> None:
    """Only a completed device job can produce a successful partial-test check."""
    check_id = needs["changes"]["outputs"]["check_id"]
    if not check_id.isdecimal() or int(check_id) <= 0:
        raise ValueError("Expected a positive check ID")
    expected = {"changes", "toolchain", JOB}
    if set(needs) != expected:
        raise ValueError("Unexpected A5 distributed dependency set")
    check_id = attempt_check(repo, check_id, os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"])
    failed = [name for name in sorted(expected) if needs[name]["result"] != "success"]
    summary = ", ".join(f"{name}: {needs[name]['result']}" for name in sorted(expected))
    update_check(
        repo,
        check_id,
        {
            "status": "completed",
            **attempt_fields(
                attempt_url(repo, os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]), summary
            ),
            "conclusion": "failure" if failed else "success",
        },
    )
    if failed:
        raise ValueError(f"A5 distributed tests did not pass: {summary}")


def main() -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    if sys.argv[1] == "prepare":
        prepare(repo, json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()))
    elif sys.argv[1] == "reopen":
        reopen(repo, json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())["workflow_run"])
    elif sys.argv[1] == "finish":
        finish(repo, json.loads(os.environ["CI_NEEDS"]))
    else:
        raise ValueError("Expected prepare or finish")


if __name__ == "__main__":
    main()
