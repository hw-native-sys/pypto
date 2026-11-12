# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Mirror trusted full-run jobs onto the PR without executing PR code.

These commit statuses are informational; only ci/full enforces the merge gate.
A separate workflow serializes monitors per PR head, including reconciliation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ci_control as control

INTERVAL_SECONDS = 60
API_RESERVE = 200
POLL_REQUESTS_PER_HOUR = 500
MONITOR_SECONDS = 5 * 60 * 60
HIDDEN_JOBS = {"changes", "toolchain", "lint-gate", "pre-commit", "clang-tidy", "full-result"}
STATUS_SOURCE = re.compile(r"Full CI ([1-9][0-9]*) attempt ([1-9][0-9]*): .+")


def latest_statuses(existing: list[dict[str, Any]]) -> dict[str, Any]:
    """Read the latest visible state per context, not a mutable check-run ID."""
    latest: dict[str, Any] = {}
    for item in existing:
        context = item.get("context", "")
        if not context.startswith("full / ") or item.get("creator", {}).get("login") != "github-actions[bot]":
            continue
        if context not in latest or item["id"] > latest[context]["id"]:
            latest[context] = item
    return latest


def job_context(name: str) -> str:
    """Keep matrix legs distinct even when a long name must be shortened."""
    context = f"full / {name}"
    if len(context) <= 100:
        return context
    suffix = hashlib.sha256(name.encode()).hexdigest()[:12]
    return f"{context[:85]}...{suffix}"


def job_status(job: dict[str, Any]) -> tuple[str, str]:
    """Commit statuses share pending for queued/running and describe the phase."""
    if job["status"] != "completed":
        return "pending", "Running" if job["status"] == "in_progress" else "Queued"
    conclusion = job.get("conclusion")
    # Skipped/neutral are display-only successes, not proof of a passing test.
    # The independent ci/full aggregator still rejects skipped required tests.
    state = "success" if conclusion in {"success", "skipped", "neutral"} else "failure"
    label = str(conclusion or "unknown result").replace("_", " ").capitalize()
    return state, label


def sync_jobs(
    repo: str, head: str, run: dict[str, Any], statuses: dict[str, Any], max_writes: int = 50
) -> bool:
    jobs = control.effective_jobs(repo, run)
    writes = 0
    for job in jobs:
        if job["name"] in HIDDEN_JOBS:
            continue
        context = job_context(job["name"])
        state, label = job_status(job)
        # Actions GITHUB_TOKEN check runs cannot select a check suite and their
        # details_url is overridden. Commit statuses preserve target_url and
        # appear independently in the PR's checks list, just like ci/full.
        payload = {
            "context": context,
            "state": state,
            "target_url": f"https://github.com/{repo}/actions/runs/{run['id']}/job/{job['id']}",
            "description": f"Full CI {run['id']} attempt {run['run_attempt']}: {label}",
        }
        # Status history is append-only, but GitHub displays its latest value
        # once per context. Avoid writes when neither state nor provenance moved.
        previous = statuses.get(context)
        if previous is not None and all(previous.get(k) == v for k, v in payload.items()):
            continue
        if writes >= max_writes:
            return False
        result = control.api(f"repos/{repo}/statuses/{head}", payload)
        if any(result.get(key) != value for key, value in payload.items()):
            raise ValueError("GitHub did not preserve the full-job status or log URL")
        statuses[context] = result
        writes += 1
    return True


def poll_budget(repo: str, attempt: int) -> tuple[int, int]:
    """Share the repository quota and reserve capacity for gate/status writers."""
    # GitHub exempts GET /rate_limit from the primary REST request limit.
    quota = control.api("rate_limit")["resources"]["core"]
    remaining = quota["remaining"] - API_RESERVE
    reset_in = max(1, quota["reset"] - time.time())
    if remaining <= 0:
        return math.ceil(reset_in) + 1, 0
    monitors = control.api(
        f"repos/{repo}/actions/workflows/ci-full-progress.yml/runs?status=in_progress&per_page=1"
    )
    active = max(1, monitors["total_count"])
    # Include per-attempt job reads, provenance/status reads and startup lookup
    # overhead. Spread a shared 500-request/hour polling budget across monitors.
    cost = 6 + attempt
    writes = max(0, min(50, remaining // active - cost))
    if writes == 0:
        # A positive quota may still be too small for a poll and one write.
        # Waiting proportionally can outlast the monitor; reset replenishes it.
        return math.ceil(reset_in) + 1, 0
    interval = math.ceil(
        max(
            INTERVAL_SECONDS,
            active * cost * 3600 / POLL_REQUESTS_PER_HOUR,
            active * cost * reset_in / remaining,
        )
    )
    return interval, writes


def monitor(repo: str, event: dict[str, Any]) -> None:
    resolved = False
    source = control.verified_event_run(repo, event, require_title=False) if "workflow_run" in event else None
    deadline = time.monotonic() + MONITOR_SECONDS
    statuses = None
    workflow = None
    errors = 0
    while time.monotonic() < deadline:
        delay = INTERVAL_SECONDS
        try:
            # Manual dispatch has no webhook identity. Resolve it within the
            # same bounded retry window as later REST reads, before using it.
            if not resolved:
                routed = control.resolved_event(repo, event)
                if routed is None:
                    return
                event = routed
                source = event["workflow_run"]
                resolved = True
                # The always() fallback may publish only an identified PR-full
                # source. Ordinary manual main CI is intentionally ignored.
                # Record this before polling so subsequent failures still get
                # a status reconciliation request.
                if os.environ.get("GITHUB_OUTPUT"):
                    control.output({"source_run": str(source["id"])})
            assert source is not None
            delay, max_writes = poll_budget(repo, source["run_attempt"])
            if max_writes == 0:
                time.sleep(min(delay, max(0, deadline - time.monotonic())))
                continue
            match = control.FULL_TITLE.fullmatch(source["display_title"])
            assert match is not None
            number, head = int(match[1]), match[2]
            run = control.api(f"repos/{repo}/actions/runs/{source['id']}")
            control.verified_event_run(repo, {**event, "workflow_run": run})
            # A delayed event may belong to an old attempt. Never project it onto a
            # later attempt or allow an unrelated run to select the PR check target.
            if any(
                run.get(key) != source.get(key)
                for key in (
                    "id",
                    "workflow_id",
                    "run_number",
                    "run_attempt",
                    "display_title",
                    "path",
                    "event",
                    "head_branch",
                )
            ):
                return
            if workflow is None:
                workflow = control.full_workflow(repo, run)
            if run["workflow_id"] != workflow["id"]:
                raise control.UntrustedRunError("Unexpected full CI workflow identity")
            pr = control.pull(repo, number)
            if pr["head"]["sha"] != head or pr["base"]["sha"] != match[3]:
                return
            if control.newer_status_exists(repo, number, head, run):
                return
            if statuses is None:
                existing = control.pages(f"repos/{repo}/commits/{head}/status", "statuses")
                statuses = latest_statuses(existing)
                # A newer projection also invalidates a stale REST snapshot.
                # Never overwrite it or create a competing older record.
                if any(
                    (published := STATUS_SOURCE.fullmatch(item.get("description") or ""))
                    and int(published[1]) == run["id"]
                    and int(published[2]) > run["run_attempt"]
                    for item in statuses.values()
                ):
                    return
            synced = sync_jobs(repo, head, run, statuses, max_writes=max_writes)
            if run["status"] == "completed" and synced:
                return
            errors = 0
        except subprocess.CalledProcessError:
            # A create may have succeeded before its response was lost. Read
            # back persisted records before retrying instead of creating again.
            statuses = None
            errors += 1
            if errors >= 5:
                raise
            delay = min(300, INTERVAL_SECONDS * 2 ** (errors - 1))
            control.report("Progress API unavailable; backing off before retrying.")
        time.sleep(min(delay, max(0, deadline - time.monotonic())))
    control.report("Live progress window ended; the completion event will reconcile final job results.")


def route(repo: str, event: dict[str, Any]) -> None:
    """Resolve a trusted PR/head key before entering the progress writer queue."""
    for attempt in range(5):
        try:
            control.route(repo, event)
            return
        except subprocess.CalledProcessError:
            if attempt == 4:
                raise
            time.sleep(min(300, INTERVAL_SECONDS * 2**attempt))


if __name__ == "__main__":
    command = route if sys.argv[1:] == ["route"] else monitor
    command(os.environ["GITHUB_REPOSITORY"], json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()))
