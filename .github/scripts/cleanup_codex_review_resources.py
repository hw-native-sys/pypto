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

LABEL = "org.pypto.codex-review."
WORKFLOW = ".github/workflows/codex-review.yml"
RESOURCE_FORMAT = (
    '[{{json .ID}},{{json (.Label "org.pypto.codex-review.run")}},'
    '{{json (.Label "org.pypto.codex-review.attempt")}}]'
)


def command(*args: str) -> str:
    """Bound host commands so reconciliation cannot hold the review lock indefinitely."""
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=30).stdout


def resources(kind: str, repository_id: str) -> dict[str, tuple[str, str]]:
    """Read ownership directly from Docker labels; old unlabelled resources are out of scope."""
    options = ("--all",) if kind == "container" else ()
    output = command(
        "docker",
        kind,
        "ls",
        "--no-trunc",
        *options,
        "--filter",
        f"label={LABEL}repository={repository_id}",
        "--format",
        RESOURCE_FORMAT,
    )
    result: dict[str, tuple[str, str]] = {}
    for line in output.splitlines():
        identity, run, attempt = json.loads(line)
        if re.fullmatch(r"[1-9][0-9]*", run) and re.fullmatch(r"[1-9][0-9]*", attempt):
            result[identity] = (run, attempt)
    return result


def remove(kind: str, identity: str) -> None:
    """A concurrent always-step may already have deleted the discovered immutable ID."""
    try:
        options = ("--force",) if kind == "container" else ()
        command("docker", kind, "rm", *options, identity)
    except subprocess.CalledProcessError:
        options = ("--all",) if kind == "container" else ()
        # Recheck existence instead of interpreting Docker's version-dependent errors.
        if identity in command("docker", kind, "ls", "--quiet", "--no-trunc", *options).split():
            raise


def completed(repo: str, repository_id: str, run: str, attempt: str) -> bool:
    """Check the exact attempt, so rerunning a workflow cannot authorize deleting its new resources."""
    record = json.loads(command("gh", "api", f"repos/{repo}/actions/runs/{run}/attempts/{attempt}"))
    if (
        record.get("id") != int(run)
        or record.get("run_attempt") != int(attempt)
        or record.get("repository", {}).get("id") != int(repository_id)
        or record.get("repository", {}).get("full_name", "").casefold() != repo.casefold()
        or record.get("path") != WORKFLOW
    ):
        raise ValueError(f"Unexpected workflow identity for run {run}, attempt {attempt}")
    return record.get("status") == "completed"


def reconcile(repo: str, repository_id: str) -> None:
    """Verify all owners before deleting containers, then networks; never touch session volumes."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not re.fullmatch(
        r"[1-9][0-9]*", repository_id
    ):
        raise ValueError("Expected an owner/repository and a positive repository ID")
    inventory = [(kind, resources(kind, repository_id)) for kind in ("container", "network")]
    owners = {owner for _, items in inventory for owner in items.values()}
    # API failure aborts before any deletion. Active or unknown attempts stay intact.
    finished = {owner for owner in owners if completed(repo, repository_id, *owner)}
    for kind, items in inventory:
        for identity, owner in items.items():
            if owner in finished:
                remove(kind, identity)
                print(f"Reclaimed review {kind} {identity} (run {owner[0]}, attempt {owner[1]})")


if __name__ == "__main__":
    reconcile(os.environ["GH_REPO"], os.environ["REPOSITORY_ID"])
