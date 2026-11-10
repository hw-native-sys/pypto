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
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any

WORKFLOW = ".github/workflows/ci.yml"
FULL_TITLE = re.compile(r"CI full PR #([1-9][0-9]*) @([0-9a-f]{40}) base:([0-9a-f]{40})")
UNFINISHED = ("queued", "in_progress", "requested", "waiting", "pending")
REQUIRED = {
    "changes",
    "toolchain",
    "lint-gate",
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
OPTIONAL = {"prebuild-wheels", "prebuild-wheels-github", "pre-commit", "clang-tidy"}
CONDITIONAL = {"docs-examples": "docs_changed", "serving-guard": "ptoas_version_changed"}


class UntrustedRunError(ValueError):
    """Reject unrelated runs without granting them permission to write statuses."""


def api(path: str, payload: dict[str, Any] | None = None, *, method: str = "POST") -> Any:
    """Use gh's token handling and pass JSON through stdin, never the shell."""
    command = ["gh", "api", "-H", "X-GitHub-Api-Version: 2022-11-28", path]
    if payload is not None:
        command += ["--method", method, "--input", "-"]
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
    # Filtered workflow-run searches stop at 1,000 results on the API side.
    # Do not treat the empty page after that cap as proof of completeness.
    page_limit = 10 if key == "workflow_runs" else 100
    for page in range(1, page_limit + 1):
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


def newer_request_exists(repo: str, number: int, head: str, run: dict[str, Any]) -> bool:
    """Only main dispatches can supersede a request, including across bases."""
    path = (
        f"repos/{repo}/actions/workflows/ci.yml/runs?event=workflow_dispatch&branch=main"
        f"&created=>={run['created_at']}&per_page=100"
    )
    for page in range(1, 11):
        batch = api(f"{path}&page={page}")["workflow_runs"]
        for candidate in batch:
            match = FULL_TITLE.fullmatch(candidate.get("display_title", ""))
            if (
                match
                and int(match[1]) == number
                and match[2] == head
                and candidate.get("head_branch") == "main"
                and candidate["run_number"] > run["run_number"]
            ):
                return True
        if len(batch) < 100:
            return False
    raise ValueError("Recent CI requests exceeded the pagination limit")


def newer_status_exists(repo: str, number: int, head: str, run: dict[str, Any]) -> bool:
    """Preserve pending commands or newer writers without searching run history."""
    statuses = pages(f"repos/{repo}/commits/{head}/status", "statuses")
    status = next((status for status in statuses if status["context"] == "ci/full"), None)
    if not status:
        return False
    if (
        status.get("state") == "pending"
        and status.get("description") == "Full CI requested; waiting for dispatch"
        and status.get("target_url") == f"https://github.com/{repo}/pull/{number}"
    ):
        return True  # An authorized command has not yet bound its replacement run.
    target = re.fullmatch(
        rf"https://github\.com/{re.escape(repo)}/actions/runs/([1-9][0-9]*)(?:/attempts/([1-9][0-9]*))?",
        status.get("target_url") or "",
    )
    if not target:
        return False
    candidate_path = f"repos/{repo}/actions/runs/{target[1]}"
    same_run = int(target[1]) == run["id"]
    if same_run:
        if target[2] is None or int(target[2]) <= run["run_attempt"]:
            return False
        # A delayed event can predate a completed native rerun. Verify the
        # exact published attempt independently of the failed live-run lookup;
        # the status URL alone cannot establish a newer trusted writer.
        candidate_path += f"/attempts/{target[2]}"
    candidate = api(candidate_path)
    if same_run and (
        candidate.get("id") != run["id"]
        or candidate.get("run_attempt") != int(target[2])
        or candidate.get("run_number") != run["run_number"]
        or candidate.get("display_title") != run["display_title"]
        or candidate.get("path") != WORKFLOW
        or candidate.get("repository", {}).get("full_name") != repo
        or candidate.get("repository", {}).get("id") != run["repository"]["id"]
        or candidate.get("head_repository", {}).get("id") != run["repository"]["id"]
    ):
        return False
    match = FULL_TITLE.fullmatch(candidate.get("display_title", ""))
    return bool(
        match
        and int(match[1]) == number
        and match[2] == head
        and candidate["workflow_id"] == run["workflow_id"]
        and candidate["event"] == "workflow_dispatch"
        and candidate["head_branch"] == "main"
        and (same_run or candidate["run_number"] > run["run_number"])
    )


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
    if pr.get("merge_commit_sha") != merge:
        raise ValueError("merge_sha is not the PR's current test merge; post /ci full again")
    commit = api(f"repos/{repo}/git/commits/{merge}")
    if [parent["sha"] for parent in commit["parents"]] != [base, head]:
        raise ValueError("Test merge does not have the authorized base and head parents")


def unfinished_request_exists(repo: str, title: str, start: datetime, end: datetime) -> bool:
    """Scan status-independent time ranges; split ranges at the API search cap."""
    if start > end:
        raise ValueError("PR creation time is later than the duplicate-check cutoff")
    windows = [(start, end)]
    while windows:
        lower, upper = windows.pop()
        created = f"{lower:%Y-%m-%dT%H:%M:%SZ}..{upper:%Y-%m-%dT%H:%M:%SZ}"
        path = (
            f"repos/{repo}/actions/workflows/ci.yml/runs?event=workflow_dispatch"
            f"&branch=main&created={created}&per_page=100"
        )
        for page in range(1, 11):
            runs = api(f"{path}&page={page}")["workflow_runs"]
            if any(
                run.get("display_title") == title
                and run.get("head_branch") == "main"
                and run.get("event") == "workflow_dispatch"
                and run["status"] in UNFINISHED
                for run in runs
            ):
                return True
            if len(runs) < 100:
                break
        else:
            # Never treat page 11 as evidence of completeness: filtered queries
            # stop at 1,000. Creation time is immutable across status transitions
            # and native reruns, so disjoint subranges preserve every candidate.
            if lower == upper:
                raise ValueError("Too many CI requests in one second; cannot verify duplicates")
            middle = lower + timedelta(seconds=int((upper - lower).total_seconds()) // 2)
            windows.extend(((lower, middle), (middle + timedelta(seconds=1), upper)))
    return False


def effective_jobs(repo: str, run: dict[str, Any]) -> list[dict[str, Any]]:
    """Overlay attempts by job name, preserving dependencies omitted by a rerun."""
    attempt = run["run_attempt"]
    if type(attempt) is not int or not 1 <= attempt <= 100:
        raise ValueError("Cannot reconstruct CI jobs: invalid or excessive attempt count")
    latest: dict[str, dict[str, Any]] = {}
    for number in range(1, attempt + 1):
        jobs = pages(f"repos/{repo}/actions/runs/{run['id']}/attempts/{number}/jobs", "jobs")
        names = [job["name"] for job in jobs]
        if len(set(names)) != len(names):
            raise ValueError("CI job names must be unique within an attempt")
        latest.update({job["name"]: job for job in jobs})
    return list(latest.values())


def matching_lint_run(repo: str, pr: dict[str, Any]) -> dict[str, Any]:
    """Reuse only the latest automatic run for this exact PR/head/base snapshot."""
    head, base = pr["head"]["sha"], pr["base"]["sha"]
    title = f"CI lint PR #{pr['number']} @{head} base:{base}"
    workflow = api(f"repos/{repo}/actions/workflows/ci.yml")
    runs = pages(
        f"repos/{repo}/actions/workflows/ci.yml/runs?event=pull_request&head_sha={head}", "workflow_runs"
    )
    candidates = [
        run
        for run in runs
        if run.get("display_title") == title
        and run.get("event") == "pull_request"
        and run.get("path") == WORKFLOW
        and run.get("workflow_id") == workflow["id"]
        and run.get("head_sha") == head
        and run.get("repository", {}).get("full_name") == repo
        and run.get("head_repository", {}).get("id") == pr["head"]["repo"]["id"]
    ]
    if not candidates:
        raise ValueError("No matching PR lint run; update the PR branch and wait for lint before /ci full")
    run = max(candidates, key=lambda item: item["run_number"])
    # Wait for the attempt to finish so not-yet-indexed rerun jobs cannot make
    # an earlier success look reusable while that check is actually rerunning.
    if run["status"] != "completed":
        raise ValueError("PR lint run is still running; wait before /ci full")
    jobs = effective_jobs(repo, run)
    for name in ("changes", "pre-commit", "clang-tidy"):
        selected = [job for job in jobs if job["name"] == name]
        if len(selected) != 1 or selected[0]["status"] != "completed":
            raise ValueError(f"PR lint job {name} is missing or still running; wait before /ci full")
        result = selected[0]["conclusion"]
        if name == "clang-tidy" and result == "skipped":
            files = pages(f"repos/{repo}/pulls/{pr['number']}/files")
            # Match the workflow's docs-only exemption, including rename sources.
            # The PR files API caps at 3,000; never accept truncated classification.
            paths = [
                value
                for file in files
                for value in (file["filename"], file.get("previous_filename"))
                if value
            ]
            if (
                files
                and len(files) < 3000
                and all(
                    path.startswith("docs/") or path.endswith(".md") or path == "mkdocs.yml" for path in paths
                )
            ):
                continue
        if result != "success":
            raise ValueError(f"PR lint job {name} did not pass; fix or rerun lint before /ci full")
    report(f"Reusing lint from https://github.com/{repo}/actions/runs/{run['id']}")
    return run


def already_passed(repo: str, pr: dict[str, Any]) -> bool:
    """Repeated comments cannot reset a verified green gate for the same revision."""
    statuses = pages(f"repos/{repo}/commits/{pr['head']['sha']}/status", "statuses")
    status = next((item for item in statuses if item["context"] == "ci/full"), {})
    target = re.fullmatch(
        rf"https://github\.com/{re.escape(repo)}/actions/runs/([1-9][0-9]*)/attempts/([1-9][0-9]*)",
        status.get("target_url") or "",
    )
    if status.get("state") != "success" or not target:
        return False
    run = api(f"repos/{repo}/actions/runs/{target[1]}")
    expected = f"CI full PR #{pr['number']} @{pr['head']['sha']} base:{pr['base']['sha']}"
    if (
        run.get("display_title") != expected
        or run.get("event") != "workflow_dispatch"
        or run.get("head_branch") != "main"
        or run.get("path") != WORKFLOW
        or run.get("repository", {}).get("full_name") != repo
        or run.get("head_repository", {}).get("id") != pr["base"]["repo"]["id"]
        or run.get("run_attempt") != int(target[2])
        or run.get("status") != "completed"
        or run.get("conclusion") == "cancelled"
    ):
        return False
    workflow = api(f"repos/{repo}/actions/workflows/ci.yml")
    if run.get("workflow_id") != workflow["id"]:
        return False
    jobs = pages(f"repos/{repo}/actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs", "jobs")
    finalizers = [job for job in jobs if job["name"] == "full-result"]
    return len(finalizers) == 1 and finalizers[0]["conclusion"] == "success"


def validated_request(repo: str, event: dict[str, Any]) -> dict[str, Any] | None:
    if (
        event.get("action") != "created"
        or "pull_request" not in event.get("issue", {})
        or event.get("comment", {}).get("body", "").strip() != "/ci full"
    ):
        return
    # Anyone who can comment may request full CI, including external contributors.
    # Validate the live PR and merge revision both before queueing and dispatching.
    number = int(event["issue"]["number"])
    pr = pull(repo, number)
    if pr.get("mergeable") is not True or not pr.get("merge_commit_sha"):
        raise ValueError("PR merge is conflicting or still being computed; retry /ci full when ready")
    head, base, merge = pr["head"]["sha"], pr["base"]["sha"], pr["merge_commit_sha"]
    verify_merge(repo, pr, head, base, merge)
    return pr


def command_route(repo: str, event: dict[str, Any]) -> None:
    """Validate the PR revision before joining the shared writer queue."""
    pr = validated_request(repo, event)
    if pr is not None:
        head = pr["head"]["sha"]
        output({"key": f"{pr['number']}-{head}", "head": head})


def dispatch(repo: str, event: dict[str, Any]) -> None:
    pr = validated_request(repo, event)
    if pr is None:
        return
    number = pr["number"]
    head, base, merge = pr["head"]["sha"], pr["base"]["sha"], pr["merge_commit_sha"]
    if os.environ.get("CI_ROUTED_HEAD") != head:
        raise ValueError("PR head changed after queue admission; post /ci full again")
    if already_passed(repo, pr):
        report("Full CI already passed for this PR revision; keeping the successful gate.")
        return
    matching_lint_run(repo, pr)
    # Metadata reads can overlap a push; revalidate before writing or dispatching.
    verify_merge(repo, pull(repo, number), head, base, merge)
    title = f"CI full PR #{number} @{head} base:{base}"
    # A full request cannot predate its PR. Fix the upper bound for pagination;
    # never filter by mutable run status, which can move between queries.
    created = datetime.strptime(pr["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    cutoff = datetime.now(timezone.utc).replace(microsecond=0)
    if unfinished_request_exists(repo, title, created, cutoff):
        report(f"Full CI for PR #{number} at {head} is already queued or running.")
        return
    # Close the old-green window before creating another run. If this write
    # fails, do not dispatch; if dispatch fails, keep the gate pending for retry.
    api(
        f"repos/{repo}/statuses/{head}",
        {
            "context": "ci/full",
            "state": "pending",
            "description": "Full CI requested; waiting for dispatch",
            "target_url": f"https://github.com/{repo}/pull/{number}",
        },
    )
    # Dispatch always chooses the trusted workflow on main, never the PR branch.
    dispatched = api(
        f"repos/{repo}/actions/workflows/ci.yml/dispatches",
        {
            "ref": "main",
            "return_run_details": True,
            "inputs": {
                "pr_number": str(number),
                "head_sha": head,
                "base_sha": base,
                "merge_sha": merge,
            },
        },
    )
    run_id = dispatched.get("workflow_run_id") if isinstance(dispatched, dict) else None
    if type(run_id) is not int or run_id <= 0:
        raise ValueError("Dispatch did not return a valid run ID; pending request remains blocked")
    # Bind ownership before releasing the shared queue, even if run-history
    # searches have not indexed the new run yet. Lost acknowledgements retain
    # the unbound marker; an older publisher must not clear it.
    api(
        f"repos/{repo}/statuses/{head}",
        {
            "context": "ci/full",
            "state": "pending",
            "description": "Full CI dispatched; waiting for results",
            "target_url": f"https://github.com/{repo}/actions/runs/{run_id}/attempts/1",
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
        matching_lint_run(repo, pr)
        checkout = inputs["merge_sha"]
    elif any(inputs.get(key) for key in ("head_sha", "base_sha", "merge_sha")):
        raise ValueError("PR revision inputs require pr_number")
    output({"full": str(full).lower(), "checkout_sha": checkout, "reuse_lint": str(bool(number)).lower()})


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


def verified_event_run(repo: str, event: dict[str, Any]) -> dict[str, Any]:
    """Validate event provenance for routing or invalidation, never to pass a gate."""
    try:
        run, workflow, repository = event["workflow_run"], event["workflow"], event["repository"]
        if (
            event["action"] not in {"requested", "in_progress", "completed"}
            or workflow["path"] != WORKFLOW
            or workflow["id"] != run["workflow_id"]
            or run["path"] != WORKFLOW
            or run["event"] != "workflow_dispatch"
            or run["head_branch"] != "main"
            or repository["full_name"] != repo
            or run["repository"]["full_name"] != repo
            or run["repository"]["id"] != repository["id"]
            or run["head_repository"]["id"] != repository["id"]
            or not FULL_TITLE.fullmatch(run["display_title"])
            or any(
                type(run[key]) is not int or run[key] <= 0
                for key in ("id", "workflow_id", "run_number", "run_attempt")
            )
            or type(repository["id"]) is not int
            or repository["id"] <= 0
        ):
            raise UntrustedRunError("Cannot validate workflow_run event provenance")
    except (KeyError, TypeError):
        raise UntrustedRunError("Incomplete workflow_run event provenance") from None
    return run


def route(repo: str, event: dict[str, Any]) -> None:
    """Reject unrelated events before they can join the status writer queue."""
    run = verified_event_run(repo, event)
    match = FULL_TITLE.fullmatch(run["display_title"])
    assert match is not None  # Already validated along with workflow identity.
    output({"key": f"{match[1]}-{match[2]}"})


def publish(repo: str, event: dict[str, Any]) -> None:
    # Success requires the live attempt: a delayed completion event may describe
    # a previous successful attempt while a native rerun is already in progress.
    live_lookup_failed = False
    try:
        run = api(f"repos/{repo}/actions/runs/{int(event['workflow_run']['id'])}")
    except (ValueError, subprocess.CalledProcessError):
        run = verified_event_run(repo, event)
        live_lookup_failed = True
    match = FULL_TITLE.fullmatch(run.get("display_title", ""))
    if not match or run["event"] != "workflow_dispatch":
        return
    if (
        run["path"] != WORKFLOW
        or run["head_branch"] != "main"
        or run["repository"]["full_name"] != repo
        or run["head_repository"]["id"] != run["repository"]["id"]
    ):
        raise ValueError("Full CI result did not originate from the trusted workflow")
    number, head, base = int(match[1]), match[2], match[3]
    # Live metadata or the validated event identifies the expected origin.
    # Lookup failures and obsolete PR revisions invalidate old green statuses;
    # untrusted workflow origins never write.
    try:
        if live_lookup_failed:
            raise ValueError("Cannot read the live CI attempt; event data cannot prove success")
        workflow = api(f"repos/{repo}/actions/workflows/ci.yml")
        if run["workflow_id"] != workflow["id"]:
            raise UntrustedRunError("Full CI result did not originate from the trusted workflow")
        if newer_status_exists(repo, number, head, run):
            report("Preserving a pending command or a verified newer CI writer.")
            return
        pr = api(f"repos/{repo}/pulls/{number}")
        if (
            pr["state"] != "open"
            or pr["head"]["sha"] != head
            or pr["base"]["ref"] != "main"
            or pr["base"]["repo"]["full_name"] != repo
        ):
            # The PR may later return to this SHA (or reopen/retarget). Keep its
            # old success from surviving a failed rerun while it was ineligible.
            # Use the guarded invalidation path to preserve a verified newer writer.
            raise ValueError("PR no longer targets the tested revision")
        # New requests supersede old runs before their first event arrives.
        # Old native reruns cannot overwrite a newer request's status either.
        if newer_request_exists(repo, number, head, run):
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
        if pr["base"]["sha"] != base:
            # Strict up-to-date rules are still needed for main advancing
            # after publication, but an already obsolete base cannot pass.
            state = "failure"
            report("The tested base is obsolete; post /ci full again against current main.")
    except UntrustedRunError:
        raise
    except (ValueError, subprocess.CalledProcessError):
        # A failed check must not preserve this run's previously green status.
        # The combined-status API returns only the latest entry per context,
        # so a known newer writer can still be protected without a run search.
        try:
            if newer_status_exists(repo, number, head, run):
                report("Preserving the verified newer request or attempt's full CI status.")
                return
        except (ValueError, subprocess.CalledProcessError):
            report("Could not verify the published full CI status either.")
        report("Cannot verify full CI metadata; invalidating ci/full. Post /ci full again.")
        state = "failure"
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
    try:
        if command == "aggregate":
            aggregate(json.loads(os.environ["CI_NEEDS"]))
            return
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        repo = os.environ["GITHUB_REPOSITORY"]
        commands = {
            "command-route": command_route,
            "dispatch": dispatch,
            "prepare": prepare,
            "route": route,
            "publish": publish,
        }
        commands[command](repo, event)
    except (ValueError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr or str(error) if isinstance(error, subprocess.CalledProcessError) else str(error)
        )
        # GitHub errors may contain repository-controlled text. Escape markup,
        # redact the token, and indent log lines so they cannot be commands.
        for key in ("GH_TOKEN", "GITHUB_TOKEN"):
            if token := os.environ.get(key):
                detail = detail.replace(token, "[REDACTED]")
        print(f"{command} failed:\n" + "\n".join(f"  {line}" for line in detail.splitlines()))
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary).open("a") as stream:
                stream.write(f"{escape(command)} failed:<pre>{escape(detail)}</pre>\n")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
