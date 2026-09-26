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
import re
import subprocess
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from publish_codex_review import github_api

WORKFLOW = ".github/workflows/codex-review.yml"
MAX_CONTEXT_BYTES = 8 * 1024 * 1024
CHECK_RUNS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      headRefOid
      commits(last: 1) {
        nodes {
          commit {
            oid
            statusCheckRollup {
              contexts(first: 100, after: $cursor) {
                nodes {
                  __typename
                  ... on CheckRun { checkSuite { workflowRun { resourcePath } } }
                }
                pageInfo { hasNextPage endCursor }
              }
            }
          }
        }
      }
    }
  }
}
"""


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


def checked_run_ids(repo: str, pr: dict) -> list[int] | None:
    """Find Actions runs attached to the current PR head; None means head drift."""
    owner, name = repo.split("/")
    cursor = None
    seen_cursors = set()
    run_ids = set()
    while True:
        response = github_api(
            "graphql",
            {
                "query": CHECK_RUNS_QUERY,
                "variables": {"owner": owner, "name": name, "number": pr["number"], "cursor": cursor},
            },
        )
        if not isinstance(response, dict) or response.get("errors"):
            raise ValueError("Failed to read PR check connections from GitHub GraphQL")
        try:
            target = response["data"]["repository"]["pullRequest"]
            commits = target["commits"]["nodes"]
            if len(commits) != 1:
                raise ValueError("Expected exactly one current PR commit")
            commit = commits[0]["commit"]
            if target["headRefOid"] != pr["head"]["sha"] or commit["oid"] != pr["head"]["sha"]:
                return None
            rollup = commit["statusCheckRollup"]
            if rollup is None:
                if cursor is not None:
                    raise ValueError("PR check connection disappeared during pagination")
                return []
            contexts = rollup["contexts"]
            for context in contexts["nodes"]:
                if context["__typename"] != "CheckRun":
                    continue
                suite = context["checkSuite"]
                run = suite["workflowRun"] if suite is not None else None
                if run is None:
                    continue
                path = run["resourcePath"]
                match = (
                    re.fullmatch(rf"/{re.escape(repo)}/actions/runs/([1-9][0-9]*)", path)
                    if isinstance(path, str)
                    else None
                )
                if match is None:
                    raise ValueError("Expected an Actions run resource path in the current repository")
                run_ids.add(int(match[1]))
            page = contexts["pageInfo"]
            if type(page["hasNextPage"]) is not bool:
                raise ValueError("Invalid PR check pagination state")
            if not page["hasNextPage"]:
                return sorted(run_ids, reverse=True)
            cursor = page["endCursor"]
            if not isinstance(cursor, str) or not cursor or cursor in seen_cursors:
                raise ValueError("Missing or repeated PR check pagination cursor")
            seen_cursors.add(cursor)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Incomplete PR check response from GitHub GraphQL") from error


def target_is_current(repo: str, pr: dict) -> bool:
    """Recheck PR state before rerunning or collecting standalone context."""
    current = github_api(f"repos/{repo}/pulls/{pr['number']}")
    return (
        current["state"] == "open"
        and not current["draft"]
        and current["head"]["sha"] == pr["head"]["sha"]
        and current["base"]["ref"] == pr["base"]["ref"]
    )


def matching_run(repo: str, run: dict, pr: dict) -> bool:
    """Require the workflow, base repository, and artifact-backed PR identity."""
    if not (
        run["event"] == "pull_request_target"
        and run["path"].split("@", 1)[0] == WORKFLOW
        and run["repository"]["full_name"] == repo
    ):
        return False
    return snapshot_matches(repo, run, pr)


def route_review(repo: str, pr: dict) -> bool:
    """Reuse an active run or rerun all jobs; return False for standalone review."""
    run_ids = checked_run_ids(repo, pr)
    if run_ids is None:
        report("PR changed while routing the command; no review was started.")
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    for run_id in run_ids:
        run = github_api(f"repos/{repo}/actions/runs/{run_id}")
        if datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")) <= cutoff:
            continue
        if not matching_run(repo, run, pr):
            continue
        # Re-read both resources immediately before deciding to mutate anything.
        if not target_is_current(repo, pr):
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
    if not target_is_current(repo, pr):
        report("PR changed while routing the command; no review was started.")
        return True
    report(
        "No verifiable current-head PR run within 30 days; starting a standalone review. "
        "This fallback does not replace existing PR Actions checks."
    )
    return False
