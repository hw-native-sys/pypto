# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise cancelled-review recovery without touching host Docker resources."""

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
LABEL = "org.pypto.codex-review."


@pytest.fixture
def cleaner():
    spec = importlib.util.spec_from_file_location(
        "cleanup_codex_review_resources", ROOT / ".github/scripts/cleanup_codex_review_resources.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def host(cleaner, monkeypatch):
    """Model immutable Docker IDs and exact GitHub attempt responses."""
    state: dict[str, Any] = {"networks": {}, "containers": {}, "runs": {}, "removed": [], "calls": []}

    def command(*args):
        state["calls"].append(args)
        if args[0] == "gh":
            endpoint = args[-1].split("/")
            assert endpoint[:5] == ["repos", "owner", "repo", "actions", "runs"]
            record = state["runs"][(endpoint[5], endpoint[7])]
            if isinstance(record, Exception):
                raise record
            return json.dumps(record)
        assert args[0] == "docker" and args[1] in {"network", "container"}
        kind, action = args[1:3]
        resources = state[f"{kind}s"]
        if action == "ls":
            if args[-1].startswith("name="):
                return "\n".join(
                    identity
                    for identity, resource in resources.items()
                    if re.search(args[-1][len("name=") :], resource["Name"])
                )
            filter_value = args[-1].removeprefix("label=")
            key, _, expected = filter_value.partition("=")
            return "\n".join(
                identity
                for identity, resource in resources.items()
                if key
                in (labels := (resource.get("Config", {}) if kind == "container" else resource)["Labels"])
                and (not expected or labels[key] == expected)
            )
        if action == "inspect":
            return json.dumps([resources[args[-1]]])
        assert action == "rm"
        identity = args[-1]
        if kind == "container":
            assert args[3] == "--force"
            for network in state["networks"].values():
                network["Containers"].pop(identity, None)
        else:
            assert not resources[identity]["Containers"], "Network still has live endpoints"
        state["removed"].append((kind, identity))
        del resources[identity]
        return ""

    monkeypatch.setattr(cleaner, "command", command)
    return state


def add_attempt(host, run="100", attempt="1", *, status="completed", legacy=False, repository="42"):
    suffix = f"{run}-{attempt}" + ("-12345" if legacy else "")
    labels = {LABEL + "run": run}
    if not legacy:
        labels |= {LABEL + "repository": repository, LABEL + "attempt": attempt}
    network_id, proxy_id = f"network-{suffix}", f"proxy-{suffix}"
    host["networks"][network_id] = {
        "Id": network_id,
        "Name": f"codex-review-{suffix}",
        "Labels": labels,
        "Containers": {proxy_id: {"Name": f"codex-proxy-{suffix}"}},
    }
    host["containers"][proxy_id] = {
        "Id": proxy_id,
        "Name": f"/codex-proxy-{suffix}",
        "Config": {"Labels": {} if legacy else labels.copy()},
    }
    host["runs"][(run, attempt)] = {
        "id": int(run),
        "run_attempt": int(attempt),
        "repository": {"id": int(repository), "full_name": "owner/repo"},
        "path": ".github/workflows/codex-review.yml",
        "status": status,
        "conclusion": "cancelled",
    }
    return network_id, proxy_id


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("conclusion", ["cancelled", "failure", "success", "timed_out"])
def test_completed_attempt_releases_proxy_before_network(cleaner, host, legacy, conclusion):
    network, proxy = add_attempt(host, legacy=legacy)
    host["runs"][("100", "1")]["conclusion"] = conclusion
    cleaner.reconcile("owner/repo", "42")
    assert host["removed"] == [("container", proxy), ("network", network)]
    # Idempotent recovery is important after a second cancellation during cleanup.
    cleaner.reconcile("owner/repo", "42")
    assert not host["networks"] and not host["containers"]


@pytest.mark.parametrize("status", ["in_progress", "queued", "waiting", "pending", None])
def test_active_or_unknown_status_retains_resources(cleaner, host, status):
    add_attempt(host, status=status)
    cleaner.reconcile("owner/repo", "42")
    assert not host["removed"]


def test_rerun_retains_active_attempt_and_cleans_old_attempt(cleaner, host):
    network, proxy = add_attempt(host)
    active_network, active_proxy = add_attempt(host, attempt="2", status="in_progress")
    cleaner.reconcile("owner/repo", "42")
    assert host["removed"] == [("container", proxy), ("network", network)]
    assert active_network in host["networks"] and active_proxy in host["containers"]


@pytest.mark.parametrize(
    "error", [subprocess.CalledProcessError(1, ["gh"]), subprocess.TimeoutExpired("gh", 30)]
)
def test_unverifiable_owner_aborts_before_any_removal(cleaner, host, error):
    add_attempt(host)
    add_attempt(host, run="101")
    host["runs"][("101", "1")] = error
    with pytest.raises(type(error)):
        cleaner.reconcile("owner/repo", "42")
    assert not host["removed"]


@pytest.mark.parametrize(
    "change",
    [
        {"id": 101},
        {"run_attempt": 2},
        {"repository": {"id": 43, "full_name": "owner/repo"}},
        {"repository": {"id": 42, "full_name": "other/repo"}},
        {"path": ".github/workflows/ci.yml"},
    ],
)
def test_wrong_github_identity_never_authorizes_removal(cleaner, host, change):
    add_attempt(host, legacy=True)
    host["runs"][("100", "1")].update(change)
    with pytest.raises(ValueError, match="Unexpected workflow identity"):
        cleaner.reconcile("owner/repo", "42")
    assert not host["removed"]


def test_other_repository_and_shared_infrastructure_are_untouched(cleaner, host):
    add_attempt(host, repository="43")
    host["networks"]["egress"] = {
        "Id": "egress",
        "Name": "pypto-codex-egress",
        "Labels": {},
        "Containers": {},
    }
    host["containers"]["relay"] = {
        "Id": "relay",
        "Name": "/pypto-codex-proxy-relay",
        "Config": {"Labels": {}},
    }
    cleaner.reconcile("owner/repo", "42")
    assert not host["removed"]
    assert not any(call[0] == "gh" for call in host["calls"])


@pytest.mark.parametrize("role", ["proxy", "agent", "session-init", "session-read", "session-write"])
def test_labelled_container_without_private_network_is_recovered(cleaner, host, role):
    network, proxy = add_attempt(host)
    del host["networks"][network]
    host["containers"][proxy]["Name"] = f"/codex-{role}-100-1"
    cleaner.reconcile("owner/repo", "42")
    assert host["removed"] == [("container", proxy)]


def test_legacy_proxy_stranded_before_network_connect_is_recovered(cleaner, host):
    network, proxy = add_attempt(host, legacy=True)
    host["networks"][network]["Containers"] = {}
    cleaner.reconcile("owner/repo", "42")
    assert host["removed"] == [("container", proxy), ("network", network)]


def test_unlabelled_proxy_without_ownership_network_is_preserved(cleaner, host):
    network, _ = add_attempt(host, legacy=True)
    del host["networks"][network]
    cleaner.reconcile("owner/repo", "42")
    assert not host["removed"]


def test_mismatched_labels_are_not_treated_as_legacy(cleaner, host):
    network, proxy = add_attempt(host, legacy=True)
    host["networks"][network]["Labels"][LABEL + "attempt"] = "2"
    host["containers"][proxy]["Config"]["Labels"][LABEL + "attempt"] = "2"
    cleaner.reconcile("owner/repo", "42")
    assert not host["removed"]


def test_legacy_network_does_not_authorize_unrelated_container(cleaner, host):
    network, proxy = add_attempt(host, legacy=True)
    host["containers"][proxy]["Name"] = "/codex-proxy-100-1-99999"
    # Docker refuses to remove a network with an unknown endpoint; it must not be forced away.
    with pytest.raises(AssertionError, match="live endpoints"):
        cleaner.reconcile("owner/repo", "42")
    assert not host["removed"]
    assert network in host["networks"] and proxy in host["containers"]


@pytest.mark.parametrize("kind", ["network", "container"])
@pytest.mark.parametrize("operation", ["inspect", "rm"])
def test_concurrent_cleanup_can_remove_discovered_resource(cleaner, monkeypatch, kind, operation):
    def disappeared(*args):
        raise subprocess.CalledProcessError(1, args, stderr=f"Error: No such {kind}: abc123")

    monkeypatch.setattr(cleaner, "command", disappeared)
    assert cleaner.docker_resource(kind, operation, "abc123") is None


@pytest.mark.parametrize("message", ["Docker daemon unavailable", "Error: No such container: other-id"])
def test_docker_failure_is_not_mistaken_for_disappearance(cleaner, monkeypatch, message):
    def failed(*args):
        raise subprocess.CalledProcessError(1, args, stderr=message)

    monkeypatch.setattr(cleaner, "command", failed)
    with pytest.raises(subprocess.CalledProcessError):
        cleaner.inspect("container", "abc123")


@pytest.fixture
def workflow():
    return yaml.safe_load((ROOT / ".github/workflows/codex-review.yml").read_text())


@pytest.mark.parametrize("docker_fails", [False, True])
def test_independent_cleanup_runs_without_review_shell(tmp_path, workflow, docker_fails):
    """Execute the real always-step against leftovers when the review shell's trap never ran."""
    steps = {step["name"]: step for step in workflow["jobs"]["review"]["steps"]}
    step = steps["Remove review containers and network"]
    assert step["if"] == "${{ always() }}"
    assert step["timeout-minutes"] == 2
    snapshots = [tmp_path / "auth", tmp_path / "patterns"]
    for snapshot in snapshots:
        snapshot.write_text("test credential snapshot")
    calls = tmp_path / "docker-calls"
    # Check the scope filters in the shell actually executed, not just YAML text.
    fake_docker = r"""
    docker() {
      printf '%s\n' "$*" >>"$CALLS"
      if [ "$1 $2" = "container ls" ] || [ "$1 $2" = "network ls" ]; then
        [[ "$*" == *"label=org.pypto.codex-review.repository=42"* ]] || return 9
        [[ "$*" == *"label=org.pypto.codex-review.run=100"* ]] || return 9
        [[ "$*" == *"label=org.pypto.codex-review.attempt=2"* ]] || return 9
        [ "$DOCKER_FAILS" = 0 ] || return 1
        if [ "$1" = container ]; then printf 'agent-id\nproxy-id\n'; else echo network-id; fi
      fi
    }
    """
    result = subprocess.run(
        ["bash", "-c", fake_docker + step["run"]],
        env=os.environ
        | {
            "CALLS": str(calls),
            "DOCKER_FAILS": str(int(docker_fails)),
            "REPOSITORY_ID": "42",
            "GITHUB_RUN_ID": "100",
            "GITHUB_RUN_ATTEMPT": "2",
            "CODEX_AUTH_SNAPSHOT": str(snapshots[0]),
            "CODEX_SECRET_PATTERNS": str(snapshots[1]),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode != 0) == docker_fails, result.stderr
    assert all(not snapshot.exists() for snapshot in snapshots)
    commands = calls.read_text().splitlines()
    if docker_fails:
        assert not any(" rm " in command for command in commands)
    else:
        assert "container rm --force agent-id proxy-id" in commands
        assert commands[-1] == "network rm network-id"
    assert not any("volume" in command or "prune" in command for command in commands)


def test_workflow_records_identities_and_loads_trusted_cleaner(workflow):
    job = workflow["jobs"]["review"]
    steps = {step["name"]: step for step in job["steps"]}
    names = list(steps)
    assert names.index("Prepare review resource identities") < names.index("Review in an isolated container")
    assert names.index("Remove review containers and network") < names.index("Remove review checkout")
    assert job["permissions"]["actions"] == "read"
    review = steps["Review in an isolated container"]["run"]
    assert 'git show "$TRUSTED_WORKFLOW_SHA:.github/scripts/cleanup_codex_review_resources.py"' in review
    assert review.index('python "$resource_cleaner"') < review.index("docker network create")
    # Every ephemeral container creation carries ownership, including session helpers.
    assert review.count('"${resource_labels[@]}"') == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
