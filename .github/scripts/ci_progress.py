# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Mirror trusted full-run jobs onto the PR without executing PR code.

These checks are informational; only ci/full enforces the merge gate. A separate
workflow serializes monitors per source run, including completion reconciliation.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import ci_control as control

INTERVAL_SECONDS = 60
API_RESERVE = 200
POLL_REQUESTS_PER_HOUR = 500
MONITOR_SECONDS = 5 * 60 * 60
HIDDEN_JOBS = {"changes", "toolchain", "lint-gate", "pre-commit", "clang-tidy", "full-result"}


def sync_jobs(
    repo: str, head: str, run: dict[str, Any], checks: dict[str, Any], max_writes: int = 50
) -> bool:
    jobs = control.effective_jobs(repo, run)
    writes = 0
    for job in jobs:
        if job["name"] in HIDDEN_JOBS:
            continue
        key = f"ci-full:{run['id']}:{run['run_attempt']}:{job['name']}"
        status = job["status"] if job["status"] in {"in_progress", "completed"} else "queued"
        payload = {
            "name": f"full / {job['name']}"[:100],
            "status": status,
            "details_url": f"https://github.com/{repo}/actions/runs/{run['id']}/job/{job['id']}",
            "external_id": key,
        }
        if status == "completed":
            payload["conclusion"] = job["conclusion"] or "failure"
        previous = checks.get(key)
        if previous is not None and all(previous.get(k) == v for k, v in payload.items()):
            continue
        if writes >= max_writes:
            return False
        if previous is None:
            result = control.api(f"repos/{repo}/check-runs", {**payload, "head_sha": head})
        else:
            result = control.api(f"repos/{repo}/check-runs/{previous['id']}", payload, method="PATCH")
        checks[key] = {**payload, "id": result["id"]}
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
    resolved = "workflow_run" not in event
    if resolved:
        routed = control.resolved_event(repo, event)
        if routed is None:
            return
        event = routed
    source = control.verified_event_run(repo, event, require_title=False)
    deadline = time.monotonic() + MONITOR_SECONDS
    checks = None
    workflow = None
    errors = 0
    while time.monotonic() < deadline:
        delay = INTERVAL_SECONDS
        try:
            delay, max_writes = poll_budget(repo, source["run_attempt"])
            if max_writes == 0:
                time.sleep(min(delay, max(0, deadline - time.monotonic())))
                continue
            if not resolved:
                routed = control.resolved_event(repo, event)
                if routed is None:
                    return
                event = routed
                source = event["workflow_run"]
                resolved = True
            match = control.FULL_TITLE.fullmatch(source["display_title"])
            assert match is not None
            number, head = int(match[1]), match[2]
            run = control.api(f"repos/{repo}/actions/runs/{source['id']}")
            control.verified_event_run(repo, {**event, "workflow_run": run})
            # A delayed event may belong to an old attempt. Never project it onto a
            # later attempt or allow an unrelated run to select the PR check target.
            for key in (
                "id",
                "workflow_id",
                "run_number",
                "run_attempt",
                "display_title",
                "path",
                "event",
                "head_branch",
            ):
                if run.get(key) != source.get(key):
                    return
            if workflow is None:
                workflow = control.api(f"repos/{repo}/actions/workflows/ci.yml")
            if run["workflow_id"] != workflow["id"]:
                raise control.UntrustedRunError("Unexpected full CI workflow identity")
            pr = control.pull(repo, number)
            if pr["head"]["sha"] != head or pr["base"]["sha"] != match[3]:
                return
            if control.newer_status_exists(repo, number, head, run):
                return
            if checks is None:
                existing = control.pages(f"repos/{repo}/commits/{head}/check-runs?filter=all", "check_runs")
                prefix = f"ci-full:{run['id']}:{run['run_attempt']}:"
                checks = {
                    item["external_id"]: item
                    for item in existing
                    if (item.get("external_id") or "").startswith(prefix)
                    and item.get("app", {}).get("slug") == "github-actions"
                }
            synced = sync_jobs(repo, head, run, checks, max_writes=max_writes)
            if run["status"] == "completed" and synced:
                return
            errors = 0
        except subprocess.CalledProcessError:
            errors += 1
            if errors >= 5:
                raise
            delay = min(300, INTERVAL_SECONDS * 2 ** (errors - 1))
            control.report("Progress API unavailable; backing off before retrying.")
        time.sleep(min(delay, max(0, deadline - time.monotonic())))
    control.report("Live progress window ended; the completion event will reconcile final job results.")


if __name__ == "__main__":
    monitor(os.environ["GITHUB_REPOSITORY"], json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()))
