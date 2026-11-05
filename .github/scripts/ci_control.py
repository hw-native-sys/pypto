# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Authorize comment CI, pin test revisions, and publish a trusted full status.

Only control jobs run this helper. The publisher reads GitHub job metadata, never
PR files, caches or artifacts. Run titles carry routing data, not authorization:
workflow identity, event, default branch and the current PR are verified too.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKFLOW = ".github/workflows/ci.yml"
FULL_TITLE = re.compile(r"CI full PR #([1-9][0-9]*) @([0-9a-f]{40})")
REQUIRED = {
    "changes",
    "toolchain",
    "pre-commit",
    "clang-tidy",
    "unit-tests",
    "examples-tests",
    "codegen-tests",
    "system-tests",
    "system-tests-direct",
    "dist-system-tests",
    "pypto-lib-compile",
    "pypto-lib-model",
    "onboard-tests-a5",
    "system-tests-a5sim",
}
OPTIONAL = {"prebuild-wheels", "prebuild-wheels-github"}
CONDITIONAL = {"docs-examples": "docs_changed", "serving-guard": "ptoas_version_changed"}


def api(path: str, payload: dict[str, Any] | None = None) -> Any:
    """Use gh's token handling and pass JSON through stdin, never the shell."""
    command = ["gh", "api", path]
    if payload is not None:
        command += ["--method", "POST", "--input", "-"]
    result = subprocess.run(
        command,
        input=json.dumps(payload) if payload is not None else None,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else None


def pages(path: str, key: str | None = None) -> list[dict[str, Any]]:
    """Fail closed rather than silently accept a truncated set of jobs or runs."""
    result: list[dict[str, Any]] = []
    separator = "&" if "?" in path else "?"
    for page in range(1, 101):
        response = api(f"{path}{separator}per_page=100&page={page}")
        batch = response[key] if key else response
        result.extend(batch)
        if len(batch) < 100:
            return result
    raise ValueError("CI metadata exceeded the pagination limit")


def report(message: str) -> None:
    print(message)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as stream:
            stream.write(f"{message}\n\n")


def output(values: dict[str, str]) -> None:
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"Multiline output is not allowed: {key}")
            stream.write(f"{key}={value}\n")


def sha(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Expected a full lowercase commit SHA")
    return value


def pull(repo: str, number: int) -> dict[str, Any]:
    pr = api(f"repos/{repo}/pulls/{number}")
    if (
        pr["state"] != "open"
        or pr["base"]["ref"] != "main"
        or pr["base"]["repo"]["full_name"] != repo
        or not pr["head"]["repo"]
    ):
        raise ValueError("CI commands require an open PR targeting this repository's main branch")
    return pr


def verify_merge(repo: str, pr: dict[str, Any], head: str, base: str, merge: str) -> None:
    """A fixed test merge preserves pull_request semantics, including forks."""
    for value in (head, base, merge):
        sha(value)
    if pr["head"]["sha"] != head or pr["base"]["sha"] != base:
        raise ValueError("PR changed; post /ci full again for the new revision")
    commit = api(f"repos/{repo}/git/commits/{merge}")
    if [parent["sha"] for parent in commit["parents"]] != [base, head]:
        raise ValueError("Test merge does not have the authorized base and head parents")


def dispatch(repo: str, event: dict[str, Any]) -> None:
    if (
        event.get("action") != "created"
        or "pull_request" not in event.get("issue", {})
        or event.get("comment", {}).get("body", "").strip() != "/ci full"
    ):
        return
    comment = event["comment"]
    actor = comment["user"]["login"]
    # author_association is not a repository permission check: a contributor may
    # have no write access, and membership alone must not spend device capacity.
    permission = api(f"repos/{repo}/collaborators/{actor}/permission")["permission"]
    if permission not in {"write", "maintain", "admin"}:
        raise ValueError("Only collaborators with write, maintain or admin permission may request full CI")
    number = int(event["issue"]["number"])
    pr = pull(repo, number)
    if pr.get("mergeable") is not True or not pr.get("merge_commit_sha"):
        raise ValueError("PR merge is conflicting or still being computed; retry /ci full when ready")
    head, base, merge = pr["head"]["sha"], pr["base"]["sha"], pr["merge_commit_sha"]
    verify_merge(repo, pr, head, base, merge)
    title = f"CI full PR #{number} @{head}"
    runs = pages(f"repos/{repo}/actions/workflows/ci.yml/runs?event=workflow_dispatch", "workflow_runs")
    if any(run["display_title"] == title and run["status"] != "completed" for run in runs):
        report(f"Full CI for PR #{number} at {head} is already queued or running.")
        return
    # Dispatch always chooses the trusted workflow on main, never the PR branch.
    api(
        f"repos/{repo}/actions/workflows/ci.yml/dispatches",
        {
            "ref": "main",
            "inputs": {
                "pr_number": str(number),
                "head_sha": head,
                "base_sha": base,
                "merge_sha": merge,
            },
        },
    )
    report(f"Requested full CI for PR #{number} at {head}; test merge {merge}.")


def prepare(repo: str, event: dict[str, Any]) -> None:
    kind = os.environ["GITHUB_EVENT_NAME"]
    inputs: dict[str, str] = event.get("inputs") or {}
    number = inputs.get("pr_number", "")
    full = kind != "pull_request" or os.environ.get("CI_COMMENT_TIERS_ENABLED") != "true"
    checkout = sha(os.environ["GITHUB_SHA"])
    if number:
        if kind != "workflow_dispatch" or os.environ["GITHUB_REF"] != "refs/heads/main":
            raise ValueError("PR full CI must use workflow_dispatch on main")
        if not re.fullmatch(r"[1-9][0-9]*", number):
            raise ValueError("Expected a positive PR number")
        pr = pull(repo, int(number))
        verify_merge(repo, pr, inputs["head_sha"], inputs["base_sha"], inputs["merge_sha"])
        checkout = inputs["merge_sha"]
    elif any(inputs.get(key) for key in ("head_sha", "base_sha", "merge_sha")):
        raise ValueError("PR revision inputs require pr_number")
    output({"full": str(full).lower(), "checkout_sha": checkout})


def aggregate(needs: dict[str, Any]) -> None:
    """Matrix job results aggregate all legs; skipped required jobs never pass."""
    expected = REQUIRED | OPTIONAL | set(CONDITIONAL)
    if set(needs) != expected:
        raise ValueError(f"Full CI dependency set changed: expected {sorted(expected)}, got {sorted(needs)}")
    failed = [name for name in REQUIRED if needs[name]["result"] != "success"]
    classification = needs["changes"].get("outputs", {})
    for name, flag in CONDITIONAL.items():
        value = classification.get(flag)
        allowed = {"success"} if value == "true" else {"skipped", "success"}
        if value not in {"true", "false"} or needs[name]["result"] not in allowed:
            failed.append(name)
    if failed:
        raise ValueError(f"Full CI did not pass required jobs: {', '.join(sorted(failed))}")
    report("All required full CI jobs passed. Wheel warm-up jobs are advisory.")


def publish(repo: str, event: dict[str, Any]) -> None:
    # The event is only a notification. Re-read the live attempt so delayed
    # completion events cannot turn an in-progress native rerun green.
    run = api(f"repos/{repo}/actions/runs/{int(event['workflow_run']['id'])}")
    match = FULL_TITLE.fullmatch(run.get("display_title", ""))
    if not match or run["event"] != "workflow_dispatch":
        return
    workflow = api(f"repos/{repo}/actions/workflows/ci.yml")
    if (
        run["path"] != WORKFLOW
        or run["workflow_id"] != workflow["id"]
        or run["head_branch"] != "main"
        or run["repository"]["full_name"] != repo
        or run["head_repository"]["id"] != run["repository"]["id"]
    ):
        raise ValueError("Full CI result did not originate from the trusted workflow")
    number, head = int(match[1]), match[2]
    pr = api(f"repos/{repo}/pulls/{number}")
    if (
        pr["state"] != "open"
        or pr["head"]["sha"] != head
        or pr["base"]["ref"] != "main"
        or pr["base"]["repo"]["full_name"] != repo
    ):
        report("Ignoring a full CI result for an obsolete or closed PR revision.")
        return
    # A newer explicit /ci full supersedes an older run, even if its first
    # workflow_run event has not been delivered yet. Old native reruns cannot
    # overwrite the newer request's status either.
    runs = pages(f"repos/{repo}/actions/workflows/ci.yml/runs?event=workflow_dispatch", "workflow_runs")
    if any(
        candidate["display_title"] == run["display_title"] and candidate["run_number"] > run["run_number"]
        for candidate in runs
    ):
        report("Ignoring a superseded full CI request.")
        return
    state = "pending"
    if run["status"] == "completed":
        # Use the current attempt's finalizer. Its needs includes successful
        # earlier jobs as well as rerun jobs; never count only the failed subset.
        jobs = pages(f"repos/{repo}/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs", "jobs")
        finalizers = [job for job in jobs if job["name"] == "full-result"]
        passed = (
            run["conclusion"] != "cancelled"
            and len(finalizers) == 1
            and finalizers[0]["conclusion"] == "success"
        )
        state = "success" if passed else "failure"
    api(
        f"repos/{repo}/statuses/{head}",
        {
            "context": "ci/full",
            "state": state,
            "description": f"Full CI: {state} (attempt {run['run_attempt']})",
            "target_url": f"https://github.com/{repo}/actions/runs/{run['id']}/attempts/{run['run_attempt']}",
        },
    )
    report(f"PR #{number} at {head}: ci/full = {state}.")


def main() -> None:
    command = sys.argv[1]
    if command == "aggregate":
        aggregate(json.loads(os.environ["CI_NEEDS"]))
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    repo = os.environ["GITHUB_REPOSITORY"]
    {"dispatch": dispatch, "prepare": prepare, "publish": publish}[command](repo, event)


if __name__ == "__main__":
    main()
