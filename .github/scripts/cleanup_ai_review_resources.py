# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reclaim review resources only after GitHub confirms their owning attempt has completed."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

from publish_ai_review import is_review_workflow

LABEL = "org.pypto.codex-review."
REMOVAL_RETRY_SECONDS = 20
RESOURCE_FORMAT = (
    '[{{json .ID}},{{json (.Label "org.pypto.codex-review.run")}},'
    '{{json (.Label "org.pypto.codex-review.attempt")}}]'
)


def command(*args: str) -> str:
    """Bound host commands so reconciliation cannot hold the review lock indefinitely."""
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=30).stdout


def resources(kind: str, repository_id: str, label: str = LABEL) -> dict[str, tuple[str, str]]:
    """Read ownership directly from Docker labels; old unlabelled resources are out of scope."""
    options = ("--all",) if kind == "container" else ()
    output = command(
        "docker",
        kind,
        "ls",
        "--no-trunc",
        *options,
        "--filter",
        f"label={label}repository={repository_id}",
        "--format",
        RESOURCE_FORMAT.replace(LABEL, label),
    )
    result: dict[str, tuple[str, str]] = {}
    for line in output.splitlines():
        identity, run, attempt = json.loads(line)
        if re.fullmatch(r"[1-9][0-9]*", run) and re.fullmatch(r"[1-9][0-9]*", attempt):
            result[identity] = (run, attempt)
    return result


def remove(kind: str, identity: str) -> None:
    """Wait for the launcher's concurrent removal or a pending network detach."""
    remove_options = ("--force",) if kind == "container" else ()
    list_options = ("--all",) if kind == "container" else ()
    deadline = time.monotonic() + REMOVAL_RETRY_SECONDS
    while True:
        try:
            command("docker", kind, "rm", *remove_options, identity)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # Docker can report a conflict while the same immutable ID is still listed.
            remaining_ids = command("docker", kind, "ls", "--quiet", "--no-trunc", *list_options)
            if identity not in remaining_ids.split():
                return
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise
            time.sleep(min(1, remaining_time))


def completed(repo: str, repository_id: str, run: str, attempt: str) -> bool:
    """Check the exact attempt, so rerunning a workflow cannot authorize deleting its new resources."""
    record = json.loads(command("gh", "api", f"repos/{repo}/actions/runs/{run}/attempts/{attempt}"))
    if (
        record.get("id") != int(run)
        or record.get("run_attempt") != int(attempt)
        or record.get("repository", {}).get("id") != int(repository_id)
        or record.get("repository", {}).get("full_name", "").casefold() != repo.casefold()
        or not is_review_workflow(record.get("path"))
    ):
        raise ValueError(f"Unexpected workflow identity for run {run}, attempt {attempt}")
    return record.get("status") == "completed"


def reconcile(
    repo: str, repository_id: str, *, provider: str = "codex", current_attempt: tuple[str, str] | None = None
) -> None:
    """Verify all owners before deleting containers, then networks; never touch session volumes."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not re.fullmatch(
        r"[1-9][0-9]*", repository_id
    ):
        raise ValueError("Expected an owner/repository and a positive repository ID")
    if provider not in {"codex", "gemini"} or (
        current_attempt is not None
        and not all(re.fullmatch(r"[1-9][0-9]*", part) for part in current_attempt)
    ):
        raise ValueError("Invalid provider or workflow attempt")
    label = f"org.pypto.{provider}-review."
    inventory = [(kind, resources(kind, repository_id, label)) for kind in ("container", "network")]
    owners = {owner for _, items in inventory for owner in items.values()}
    # API failure aborts before any deletion. Active or unknown attempts stay intact.
    finished = (
        {current_attempt}
        if current_attempt
        else {owner for owner in owners if completed(repo, repository_id, *owner)}
    )
    for kind, items in inventory:
        for identity, owner in items.items():
            if owner in finished:
                remove(kind, identity)
                print(f"Reclaimed review {kind} {identity} (run {owner[0]}, attempt {owner[1]})")


if __name__ == "__main__":
    if sys.argv[1:] == ["--current-gemini"]:
        reconcile(
            os.environ["GITHUB_REPOSITORY"],
            os.environ["GITHUB_REPOSITORY_ID"],
            provider="gemini",
            current_attempt=(os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]),
        )
    elif len(sys.argv) == 1:
        reconcile(os.environ["GH_REPO"], os.environ["REPOSITORY_ID"])
    else:
        sys.exit("Unknown review cleanup command")
