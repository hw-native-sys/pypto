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
    verify_tested_base,
)

COMMAND = "/ci a5-distributed"
JOB = "onboard-tests-a5-distributed"
# Legacy checks have no PR prefix. An attempt check adds :attempt:original_id.
CHECK_OWNER = re.compile(r"(?:a5-pr-(?P<pr>[1-9][0-9]*):)?(?P<run>[1-9][0-9]*)(?::[1-9][0-9]*:[1-9][0-9]*)?")


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


def request_number(run: dict[str, Any]) -> str:
    if (
        run["path"] != ".github/workflows/ci-a5-distributed.yml"
        or run["event"] != "issue_comment"
        or run["head_branch"] != "main"
    ):
        return ""
    match = re.fullmatch(r"A5 distributed PR #([1-9][0-9]*) \(comment [1-9][0-9]*\)", run["display_title"])
    return match[1] if match else ""


def reusable_request(repo: str, number: int, checks: list[dict[str, Any]], run_id: str) -> str | None:
    """Look up the owning workflow so cancelled or failed publishers never block a retry."""
    owners: dict[str, list[dict[str, Any]]] = {}
    for check in checks:
        owner = CHECK_OWNER.fullmatch(check["external_id"] or "")
        if (
            owner
            and owner["pr"] in (None, str(number))
            and owner["run"] != run_id
            and check["app"]["slug"] == "github-actions"
        ):
            owners.setdefault(owner["run"], []).append(check)
    reusable = None
    for owner_id, owned in owners.items():
        try:
            run = api(f"repos/{repo}/actions/runs/{owner_id}")
        except subprocess.CalledProcessError as error:
            if not re.search(r"\bHTTP 404\b", error.stderr or ""):
                raise
            # Only scoped checks on this head carry enough identity after deletion.
            scoped = [check for check in owned if check["external_id"].startswith(f"a5-pr-{number}:")]
            retire_unfinished(repo, scoped)
            continue
        if request_number(run) != str(number):
            continue
        url = attempt_url(repo, owner_id, str(run["run_attempt"]))
        if run["status"] != "completed" or not run["conclusion"]:
            reusable = reusable or url
        elif run["conclusion"] == "success":
            if any(check["conclusion"] == "success" and same_attempt(check, url) for check in owned):
                reusable = reusable or url
        else:
            retire_unfinished(repo, owned)
    return reusable


def retire_unfinished(repo: str, checks: list[dict[str, Any]]) -> None:
    for check in checks:
        if check["status"] != "completed":
            update_check(repo, str(check["id"]), {"status": "completed", "conclusion": "cancelled"})


def same_attempt(check: dict[str, Any], url: str) -> bool:
    return (check["output"]["summary"] or "").startswith(f"[Workflow attempt]({url})\n")


def attempt_check(repo: str, original_id: str, run_id: str, attempt: str) -> str:
    """Select or create the check for a rerun under the shared writer queue."""
    if attempt == "1":
        return original_id
    original = api(f"repos/{repo}/check-runs/{original_id}")
    owner_key = original["external_id"]
    if original["name"] != "ci/a5-distributed" or not re.fullmatch(
        rf"(?:a5-pr-[1-9][0-9]*:)?{run_id}", owner_key or ""
    ):
        raise ValueError("Unexpected A5 check ownership")
    url = attempt_url(repo, run_id, attempt)
    head = sha(original["head_sha"])
    key = f"{owner_key}:{attempt}:{original_id}"
    family = [
        check
        for check in head_checks(repo, head)
        if check["app"]["id"] == original["app"]["id"]
        and re.fullmatch(rf"{owner_key}:[1-9][0-9]*:{original_id}", check["external_id"] or "")
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


def rerun_device_check(repo: str, run: dict[str, Any]) -> str | None:
    """Return an active device check ID, an empty handoff, or None while jobs are unlisted."""
    live = api(f"repos/{repo}/actions/runs/{run['id']}")
    attempt = run["run_attempt"]
    if live["status"] == "completed" or live["run_attempt"] != attempt:
        return ""
    jobs = pages(f"repos/{repo}/actions/runs/{run['id']}/attempts/{attempt}/jobs", "jobs")
    if any(job["name"] == "changes" and job["status"] != "completed" for job in jobs):
        return ""  # Full reruns publish progress in prepare.
    for job in jobs:
        match = re.fullmatch(r"onboard-tests-a5-distributed / check ([1-9][0-9]*)", job["name"])
        if match and job["run_attempt"] == attempt and job["status"] != "completed":
            return match[1]
    if any(job["name"] == "result" and job["run_attempt"] == attempt for job in jobs):
        return ""  # The publisher can now finish or recover its own check.
    return None


def route_rerun(repo: str, run: dict[str, Any]) -> None:
    number = request_number(run)
    if not number or run["run_attempt"] < 2:
        return
    # Wait for job metadata in the read-only job, outside the PR writer queue.
    for _ in range(12):
        check = rerun_device_check(repo, run)
        if check is not None:
            if check:
                output({"number": number})
            return
        time.sleep(5)
    raise ValueError("Could not identify the A5 rerun job")


def reopen(repo: str, run: dict[str, Any]) -> None:
    """Recheck the current attempt once after acquiring the shared writer lock."""
    if request_number(run) and run["run_attempt"] >= 2:
        check = rerun_device_check(repo, run)
        if check:
            attempt_check(repo, check, str(run["id"]), str(run["run_attempt"]))


def pinned_merge(repo: str, number: int, head: str, live_base: str) -> tuple[str, str]:
    """Return the GitHub test merge and its actual first parent for this request."""
    deadline = time.monotonic() + MERGE_WAIT_SECONDS
    while True:
        pr = pull(repo, number)
        if pr["head"]["sha"] != head or current_base(repo) != live_base:
            raise ValueError(f"PR head or base changed while waiting; post {COMMAND} again")
        if pr.get("mergeable") is False:
            raise ValueError("PR has merge conflicts; update the PR branch before retrying")
        if pr.get("mergeable") is True and pr.get("merge_commit_sha"):
            merge = sha(pr["merge_commit_sha"])
            commit = api(f"repos/{repo}/git/commits/{merge}")
            parents = commit["parents"]
            if len(parents) == 2 and parents[1]["sha"] == head:
                tested_base = sha(parents[0]["sha"])
                verify_tested_base(repo, tested_base, live_base, COMMAND)
                break
        # GitHub may still be computing the merge for this head.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(f"GitHub test merge is not ready; retry {COMMAND} later")
        time.sleep(min(MERGE_POLL_SECONDS, remaining))
    # A push can land while we read the merge parents and compare them with main.
    latest = pull(repo, number)
    if (
        latest["head"]["sha"] != head
        or latest.get("merge_commit_sha") != merge
        or latest.get("mergeable") is not True
        or current_base(repo) != live_base
    ):
        raise ValueError(f"PR or its test merge changed during validation. Post {COMMAND} again")
    return merge, tested_base


def prepare(repo: str, event: dict[str, Any]) -> None:
    """Accept an exact new PR command and freeze its current head/base/merge."""
    if (
        event.get("action") != "created"
        or "pull_request" not in event.get("issue", {})
        or event.get("comment", {}).get("body", "").strip() != COMMAND
    ):
        return
    number = int(event["issue"]["number"])
    pr = pull(repo, number)
    head, live_base = sha(pr["head"]["sha"]), current_base(repo)
    lint = matching_lint_run(repo, pr, COMMAND)
    if lint["conclusion"] != "success":
        raise ValueError(f"PR lint workflow did not pass; fix or rerun lint before {COMMAND}")
    run_id = os.environ["GITHUB_RUN_ID"]
    checks = head_checks(repo, head)
    existing = reusable_request(repo, number, checks, run_id)
    if existing:
        # Reuse the original check only while this PR still has an eligible merge.
        pinned_merge(repo, number, head, live_base)
        report(f"Reusing the A5 distributed run for PR #{number} at {head}. [Run]({existing})")
        return
    # Open the check on the linted head before waiting for the merge so PR Checks
    # can show admission failures even when no device job starts.
    attempt = os.environ["GITHUB_RUN_ATTEMPT"]
    url = attempt_url(repo, run_id, attempt)
    owner_key = f"a5-pr-{number}:{run_id}"
    for previous in checks:
        if previous["external_id"] in (run_id, owner_key) and previous["app"]["slug"] == "github-actions":
            # Keep the original ID in job outputs so finish can find the whole
            # family, but publish the active attempt before scheduling any jobs.
            check_id = str(previous["id"])
            active_id = attempt_check(repo, check_id, run_id, attempt)
            break
    else:
        check = api(
            f"repos/{repo}/check-runs",
            {
                "name": "ci/a5-distributed",
                "head_sha": head,
                "external_id": owner_key,
                "status": "in_progress",
                **attempt_fields(url, "Preparing the distributed check."),
            },
        )
        if type(check.get("id")) is not int or check["id"] <= 0:
            raise ValueError("GitHub did not return a valid check ID")
        check_id = active_id = str(check["id"])
    try:
        merge, tested_base = pinned_merge(repo, number, head, live_base)
        output({"checkout_sha": merge, "check_id": check_id})
    except Exception as error:
        # Without the outputs the result job cannot close this attempt's check.
        try:
            update_check(
                repo,
                active_id,
                {
                    "status": "completed",
                    "conclusion": "failure",
                    **attempt_fields(url, f"Could not schedule A5 distributed tests. {error}"),
                },
            )
        except subprocess.CalledProcessError as cleanup_error:
            report(f"Could not complete check {active_id}: {cleanup_error}")
        raise
    report(
        f"A5 distributed tests for PR #{number}: head {head}, base {tested_base}, merge {merge}. [Run]({url})"
    )


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
    elif sys.argv[1] == "route-rerun":
        run = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())["workflow_run"]
        route_rerun(repo, run)
    elif sys.argv[1] == "reopen":
        reopen(repo, json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())["workflow_run"])
    elif sys.argv[1] == "finish":
        finish(repo, json.loads(os.environ["CI_NEEDS"]))
    else:
        raise ValueError("Expected prepare, route-rerun, reopen or finish")


if __name__ == "__main__":
    main()
