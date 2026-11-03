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
from typing import Any

LABEL = "org.pypto.codex-review."
WORKFLOW = ".github/workflows/codex-review.yml"
SUFFIX = r"(?P<run>[1-9][0-9]*)-(?P<attempt>[1-9][0-9]*)(?:-[0-9]+)?"
CONTAINER = re.compile(rf"codex-(?:agent|proxy|session-init|session-read|session-write)-{SUFFIX}")
NETWORK = re.compile(rf"codex-review-{SUFFIX}")


def command(*args: str) -> str:
    """Bound host commands so reconciliation cannot hold the review lock indefinitely."""
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=30).stdout


def docker_resource(kind: str, operation: str, identity: str, *options: str) -> str | None:
    """Ignore only confirmed disappearance during concurrent per-run cleanup."""
    try:
        return command("docker", kind, operation, *options, identity)
    except subprocess.CalledProcessError as error:
        message = (error.stderr or "").strip().lower()
        missing = {
            f"error: no such {kind}: {identity}",
            f"error response from daemon: no such {kind}: {identity}",
            f"error response from daemon: {kind} {identity} not found",
        }
        if message in missing:
            return None
        raise


def inspect(kind: str, identity: str) -> dict[str, Any] | None:
    """Inspect a single immutable Docker ID, never a reusable resource name."""
    output = docker_resource(kind, "inspect", identity)
    if output is None:
        return None
    records = json.loads(output)
    if len(records) != 1 or records[0]["Id"] != identity:
        raise ValueError(f"Unexpected Docker {kind} identity for {identity}")
    return records[0]


def owner(
    resource: dict[str, Any], kind: str, repository_id: str, *, legacy: bool = False
) -> tuple[str, str] | None:
    """Require matching names and ownership labels; old networks carried only a run label."""
    pattern = NETWORK if kind == "network" else CONTAINER
    match = pattern.fullmatch(resource["Name"].lstrip("/"))
    if match is None:
        return None
    labels: dict[str, str] = (resource.get("Config", {}) if kind == "container" else resource).get(
        "Labels"
    ) or {}
    run, attempt = match.group("run", "attempt")
    expected = {"repository": repository_id, "run": run, "attempt": attempt}
    if all(labels.get(LABEL + key) == value for key, value in expected.items()):
        return run, attempt
    # Legacy containers must also match a validated legacy review network suffix.
    # Never interpret partially present modern labels as legacy ownership.
    if legacy and not any(LABEL + key in labels for key in ("repository", "attempt")):
        if kind == "network" and labels.get(LABEL + "run") == run:
            return run, attempt
        if kind == "container" and not any(key.startswith(LABEL) for key in labels):
            return run, attempt
    return None


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
    """Collect and validate before deleting; preserve sessions, shared infrastructure and unknown owners."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not re.fullmatch(
        r"[1-9][0-9]*", repository_id
    ):
        raise ValueError("Expected an owner/repository and a positive repository ID")

    networks: dict[str, tuple[str, str]] = {}
    containers: dict[str, tuple[str, str]] = {}
    legacy_suffixes: set[str] = set()
    network_ids = command("docker", "network", "ls", "--quiet", "--no-trunc", "--filter", f"label={LABEL}run")
    for identity in network_ids.split():
        network = inspect("network", identity)
        if network is None:
            continue
        ownership = owner(network, "network", repository_id, legacy=True)
        if ownership is None:
            continue
        networks[identity] = ownership
        if LABEL + "repository" not in (network.get("Labels") or {}):
            legacy_suffixes.add(network["Name"][len("codex-review-") :])

    # Names narrow discovery, but never authorize deletion by themselves. Modern
    # labels cover containers stranded before network attachment. For old proxies,
    # require an exact matching run-labelled private network even when connect was
    # interrupted: a detached proxy can already occupy an egress address then.
    container_ids = command(
        "docker", "container", "ls", "--all", "--quiet", "--no-trunc", "--filter", "name=^/codex-"
    )
    for identity in container_ids.split():
        container = inspect("container", identity)
        if container is None:
            continue
        name = container["Name"].lstrip("/")
        legacy = any(
            name.startswith(f"codex-{role}-") and name[len(f"codex-{role}-") :] in legacy_suffixes
            for role in ("proxy", "agent")
        )
        ownership = owner(container, "container", repository_id, legacy=legacy)
        if ownership is not None:
            containers[identity] = ownership

    # A failed/ambiguous API response aborts before *any* deletion. A cancelled
    # conclusion is not enough: GitHub must mark the owning attempt completed.
    finished = {
        ownership
        for ownership in set(networks.values()) | set(containers.values())
        if completed(repo, repository_id, *ownership)
    }
    for kind, resources in (("container", containers), ("network", networks)):
        for identity, ownership in resources.items():
            if ownership in finished:
                args = ("--force",) if kind == "container" else ()
                if docker_resource(kind, "rm", identity, *args) is None:
                    continue
                print(f"Removed stale review {kind} {identity} (run {ownership[0]}, attempt {ownership[1]})")


if __name__ == "__main__":
    reconcile(os.environ["GH_REPO"], os.environ["REPOSITORY_ID"])
