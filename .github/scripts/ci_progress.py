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
import os
import time
from pathlib import Path
from typing import Any

import ci_control as control

INTERVAL_SECONDS = 30
MONITOR_SECONDS = 5 * 60 * 60
HIDDEN_JOBS = {"changes", "toolchain", "lint-gate", "pre-commit", "clang-tidy", "full-result"}


def sync_jobs(repo: str, head: str, run: dict[str, Any], checks: dict[str, Any]) -> None:
    # filter=latest includes previously successful jobs retained by native reruns.
    jobs = control.pages(f"repos/{repo}/actions/runs/{run['id']}/jobs?filter=latest", "jobs")
    for job in jobs:
        if job["name"] in HIDDEN_JOBS:
            continue
        key = f"ci-full:{run['id']}:{run['run_attempt']}:{job['id']}"
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
        if previous is None:
            result = control.api(f"repos/{repo}/check-runs", {**payload, "head_sha": head})
        else:
            result = control.api(f"repos/{repo}/check-runs/{previous['id']}", payload, method="PATCH")
        checks[key] = {**payload, "id": result["id"]}


def monitor(repo: str, event: dict[str, Any]) -> None:
    source = control.verified_event_run(repo, event)
    match = control.FULL_TITLE.fullmatch(source["display_title"])
    assert match is not None
    number, head = int(match[1]), match[2]
    deadline = time.monotonic() + MONITOR_SECONDS
    checks = None
    while True:
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
        sync_jobs(repo, head, run, checks)
        if run["status"] == "completed":
            return
        if time.monotonic() >= deadline:
            control.report(
                "Live progress window ended; the completion event will reconcile final job results."
            )
            return
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    monitor(os.environ["GITHUB_REPOSITORY"], json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()))
