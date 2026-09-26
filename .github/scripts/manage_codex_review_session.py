# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Manage PR session volumes from trusted host code while the caller holds the auth lock."""

import json
import os
import re
import subprocess
import sys

from publish_codex_review import github_api, pull_request_endpoint

EPOCH_LABEL = "org.pypto.codex-review.close-epoch"


def session_state(repo: str, number: str) -> tuple[dict, str]:
    """Read eligibility and the latest close event together, including merge-only closures."""
    pull_request_endpoint(repo, number)
    owner, name = repo.split("/")
    response = github_api(
        "graphql",
        {
            "query": """query($owner:String!, $name:String!, $number:Int!) {
              repository(owner:$owner, name:$name) { nameWithOwner
                pullRequest(number:$number) { number state isDraft headRefOid
                  timelineItems(last:1, itemTypes:[CLOSED_EVENT]) {
                    nodes { ... on ClosedEvent { id } }
                  }
                }
              }
            }""",
            "variables": {"owner": owner, "name": name, "number": int(number)},
        },
    )
    if response.get("errors"):
        raise ValueError("GitHub session-state query failed")
    repository = response["data"]["repository"]
    pr = repository["pullRequest"]
    if (
        repository["nameWithOwner"].casefold() != repo.casefold()
        or pr["number"] != int(number)
        or pr["state"] not in {"OPEN", "CLOSED", "MERGED"}
        or type(pr["isDraft"]) is not bool
        or not re.fullmatch(r"[0-9a-f]{40}", pr["headRefOid"])
    ):
        raise ValueError("Unexpected repository or PR session state")
    events = pr["timelineItems"]["nodes"]
    if not isinstance(events, list) or len(events) > 1:
        raise ValueError("Expected at most one latest close event")
    epoch = "initial"
    if events:
        event_id = events[0]["id"]
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Missing latest close-event identity")
        epoch = f"closed:{event_id}"
    return pr, epoch


def volume_labels(name: str) -> dict | None:
    """Inspect only the requested volume; distinguish absence from Docker failures."""
    result = subprocess.run(
        ["docker", "volume", "inspect", name], check=False, capture_output=True, text=True
    )
    if result.returncode and "no such volume" in result.stderr.lower() and name in result.stderr:
        return None
    result.check_returncode()
    volumes = json.loads(result.stdout)
    if not isinstance(volumes, list) or len(volumes) != 1 or volumes[0]["Name"] != name:
        raise ValueError("Unexpected Docker volume identity")
    labels = volumes[0].get("Labels")
    if labels is not None and not isinstance(labels, dict):
        raise ValueError("Unexpected Docker volume labels")
    return labels or {}


def manage_session(repo: str, number: str, repository_id: str, mode: str, head: str | None = None) -> str:
    """Reset stale epochs or remove closed sessions; caller holds the shared lock throughout."""
    if not re.fullmatch(r"[1-9][0-9]*", repository_id) or mode not in {"review", "cleanup"}:
        raise ValueError("Invalid repository ID or session-management mode")
    if mode == "review" and (head is None or not re.fullmatch(r"[0-9a-f]{40}", head)):
        raise ValueError("Review session management requires an exact head SHA")
    pr, epoch = session_state(repo, number)
    closed = pr["state"] in {"CLOSED", "MERGED"}
    if mode == "review" and (closed or pr["isDraft"] or pr["headRefOid"] != head):
        return "skipped"

    session_volume = f"pypto-codex-sessions-{repository_id}-{number}"
    names = (session_volume, f"{session_volume}-index")
    labels = {name: volume_labels(name) for name in names}
    current = all(value is not None and value.get(EPOCH_LABEL) == epoch for value in labels.values())
    if mode == "cleanup" and not closed and current:
        return "retained"
    if closed or not current:
        for name, value in labels.items():
            if value is not None:
                subprocess.run(["docker", "volume", "rm", name], check=True, capture_output=True, text=True)
    if mode == "cleanup":
        return "removed"
    if not current:
        for name in names:
            subprocess.run(
                ["docker", "volume", "create", "--label", f"{EPOCH_LABEL}={epoch}", name],
                check=True,
                capture_output=True,
                text=True,
            )
    return "ready"


if __name__ == "__main__":
    print(
        manage_session(
            os.environ["GH_REPO"],
            os.environ["PR_NUMBER"],
            os.environ["REPOSITORY_ID"],
            sys.argv[1],
            os.environ.get("HEAD_SHA"),
        )
    )
