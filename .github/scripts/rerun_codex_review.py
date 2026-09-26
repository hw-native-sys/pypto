# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Route authorized review commands back to the PR's original Actions run."""

from __future__ import annotations

import io
import json
import os
import subprocess
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from publish_codex_review import github_api

WORKFLOW = ".github/workflows/codex-review.yml"
MAX_CONTEXT_BYTES = 8 * 1024 * 1024


def report(message: str) -> None:
    """Expose routing decisions in logs and the command run's summary."""
    print(message)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a") as stream:
            stream.write(f"{message}\n\n")


def snapshot_matches(repo: str, run: dict, pr: dict) -> bool:
    """Verify run identity using a trusted manifest or a legacy discussion snapshot."""
    # Fork pull_request_target runs can have an empty pull_requests array.
    # A branch/SHA alone cannot distinguish PRs against different base branches.
    page = 1
    artifacts = []
    while True:
        batch = github_api(f"repos/{repo}/actions/runs/{run['id']}/artifacts?per_page=100&page={page}")[
            "artifacts"
        ]
        artifacts.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    candidates = []
    for artifact in artifacts:
        for kind in ("codex-target", "codex-context"):
            prefix = f"{kind}-{run['id']}-"
            if (
                artifact["name"].startswith(prefix)
                and artifact["name"][len(prefix) :].isdigit()
                and not artifact["expired"]
            ):
                candidates.append((int(artifact["name"][len(prefix) :]), kind == "codex-target", artifact))
    if not candidates:
        return False
    _, is_target, artifact = max(candidates, key=lambda item: item[:2])
    filename = "review-target.json" if is_target else "discussion.json"
    if not 0 < artifact["size_in_bytes"] <= MAX_CONTEXT_BYTES:
        raise ValueError("Review routing snapshot exceeds the size limit")
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/actions/artifacts/{artifact['id']}/zip"],
        capture_output=True,
        check=True,
    )
    with zipfile.ZipFile(io.BytesIO(result.stdout)) as archive:
        if archive.namelist() != [filename]:
            raise ValueError(f"Review routing artifact must contain only {filename}")
        if archive.getinfo(filename).file_size > MAX_CONTEXT_BYTES:
            raise ValueError("Review routing snapshot exceeds the size limit")
        snapshot = json.loads(archive.read(filename))
    return all(
        snapshot.get(key) == value
        for key, value in {
            "repository": repo,
            "number": pr["number"],
            "head": pr["head"]["sha"],
            "base_ref": pr["base"]["ref"],
        }.items()
    )


def matching_run(repo: str, run: dict, pr: dict) -> bool:
    """Require exact workflow, repository, source branch, head, and PR identity."""
    if not (
        run["event"] == "pull_request_target"
        and run["path"] == WORKFLOW
        and run["repository"]["full_name"] == repo
        and run["head_sha"] == pr["head"]["sha"]
        and run["head_branch"] == pr["head"]["ref"]
        and run["head_repository"]["id"] == pr["head"]["repo"]["id"]
    ):
        return False
    return snapshot_matches(repo, run, pr)


def route_review(repo: str, pr: dict) -> bool:
    """Reuse an active run or rerun all jobs; return False for standalone review."""
    runs = []
    page = 1
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    while True:
        response = github_api(
            f"repos/{repo}/actions/workflows/codex-review.yml/runs"
            f"?event=pull_request_target&head_sha={pr['head']['sha']}&per_page=100&page={page}"
        )
        runs.extend(response["workflow_runs"])
        if len(runs) >= response["total_count"] or not response["workflow_runs"]:
            break
        if len(runs) >= 1000:
            raise ValueError("Too many matching workflow runs; refusing an incomplete routing search")
        page += 1
    for run in sorted(runs, key=lambda item: item["id"], reverse=True):
        if datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")) <= cutoff:
            continue
        if not matching_run(repo, run, pr):
            continue
        # Re-read both resources immediately before deciding to mutate anything.
        current = github_api(f"repos/{repo}/pulls/{pr['number']}")
        if (
            current["state"] != "open"
            or current["draft"]
            or current["head"]["sha"] != pr["head"]["sha"]
            or current["base"]["ref"] != pr["base"]["ref"]
        ):
            report("PR changed while routing the command; no review was started.")
            return True
        latest = github_api(f"repos/{repo}/actions/runs/{run['id']}")
        url = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repo}/actions/runs/{run['id']}"
        if latest["status"] != "completed":
            report(f"Review already queued or running: {url}")
            return True
        token = os.environ.get("CODEX_REVIEW_RERUN_TOKEN")
        if not token:
            raise ValueError("Set CODEX_REVIEW_RERUN_TOKEN with repository Actions: write to rerun PR checks")
        # The rerun endpoint returns an empty 201 response, not JSON. Use a
        # dedicated token so workflow-trigger suppression cannot swallow the run.
        subprocess.run(
            ["gh", "api", "--method", "POST", f"repos/{repo}/actions/runs/{run['id']}/rerun"],
            env={**os.environ, "GH_TOKEN": token},
            capture_output=True,
            text=True,
            check=True,
        )
        report(f"Requested all jobs again on the PR's existing review run: {url}")
        return True
    report(
        "No verifiable current-head PR run within 30 days; starting a standalone review. "
        "This fallback does not replace existing PR Actions checks."
    )
    return False
