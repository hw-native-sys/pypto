# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify session lifecycle ordering with trusted API and Docker metadata fixtures."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

HEAD = "a" * 40
NAMES = ("pypto-codex-sessions-42-12", "pypto-codex-sessions-42-12-index")


@pytest.fixture
def manager(monkeypatch):
    """Load the host helper without executing compiler code or requiring Docker."""
    scripts = Path(__file__).resolve().parents[3] / ".github/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "manage_codex_review_session", scripts / "manage_codex_review_session.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def state(manager, monkeypatch):
    """Model serialized host operations; only the exact PR's two volumes may be touched."""
    state = {
        "pr": {
            "number": 12,
            "state": "OPEN",
            "isDraft": False,
            "headRefOid": HEAD,
            "timelineItems": {"nodes": []},
        },
        "volumes": {},
        "commands": [],
        "fail": None,
    }

    def api(endpoint, payload):
        assert endpoint == "graphql"
        assert payload["variables"] == {"owner": "owner", "name": "repo", "number": 12}
        assert "timelineItems(last:1, itemTypes:[CLOSED_EVENT])" in payload["query"]
        return {"data": {"repository": {"nameWithOwner": "owner/repo", "pullRequest": state["pr"]}}}

    def docker(command, *, check=False, **kwargs):
        assert command[:2] == ["docker", "volume"]
        action, name = command[2], command[-1]
        assert name in NAMES
        state["commands"].append((action, name))
        code, output, error = 0, "", ""
        if state["fail"] == action:
            code, error = 1, "Docker daemon unavailable"
        elif action == "inspect":
            if name in state["volumes"]:
                output = json.dumps([{"Name": name, "Labels": state["volumes"][name]}])
            else:
                code, output, error = 1, "[]", f"Error: No such volume: {name}"
        elif action == "rm":
            del state["volumes"][name]
        elif action == "create":
            assert command[3] == "--label"
            key, value = command[4].split("=", 1)
            state["volumes"][name] = {key: value}
        else:
            pytest.fail(f"Unexpected Docker operation: {action}")
        result = subprocess.CompletedProcess(command, code, output, error)
        if check:
            result.check_returncode()
        return result

    monkeypatch.setattr(manager, "github_api", api)
    monkeypatch.setattr(manager.subprocess, "run", docker)
    return state


def populate(manager, state, epoch="initial"):
    """Install a complete pair with a trusted generation label."""
    state["volumes"] = {name: {manager.EPOCH_LABEL: epoch} for name in NAMES}


@pytest.mark.parametrize("closed", ["CLOSED", "MERGED"])
def test_late_review_of_closed_pr_creates_nothing(manager, state, closed):
    state["pr"]["state"] = closed
    assert manager.manage_session("owner/repo", "12", "42", "review", HEAD) == "skipped"
    assert not state["commands"] and not state["volumes"]


@pytest.mark.parametrize("change", ["draft", "head"])
def test_stale_or_draft_review_creates_nothing(manager, state, change):
    if change == "draft":
        state["pr"]["isDraft"] = True
    else:
        state["pr"]["headRefOid"] = "b" * 40
    assert manager.manage_session("owner/repo", "12", "42", "review", HEAD) == "skipped"
    assert not state["commands"]


def test_new_review_reuses_only_matching_epoch(manager, state):
    assert manager.manage_session("owner/repo", "12", "42", "review", HEAD) == "ready"
    assert all(labels == {manager.EPOCH_LABEL: "initial"} for labels in state["volumes"].values())
    state["commands"].clear()
    assert manager.manage_session("owner/repo", "12", "42", "review", HEAD) == "ready"
    assert all(action == "inspect" for action, _ in state["commands"])


@pytest.mark.parametrize("old", ["legacy", "older_epoch", "missing_session", "missing_index", "mixed"])
def test_review_resets_unknown_stale_or_partial_pairs(manager, state, old):
    populate(manager, state)
    if old == "legacy":
        state["volumes"] = {name: {} for name in NAMES}
    elif old == "older_epoch":
        state["pr"]["timelineItems"]["nodes"] = [{"id": "CLOSE_1"}]
    elif old.startswith("missing"):
        del state["volumes"][NAMES[old == "missing_index"]]
    else:
        state["volumes"][NAMES[0]][manager.EPOCH_LABEL] = "closed:old"
    assert manager.manage_session("owner/repo", "12", "42", "review", HEAD) == "ready"
    expected = "closed:CLOSE_1" if old == "older_epoch" else "initial"
    assert state["volumes"] == {name: {manager.EPOCH_LABEL: expected} for name in NAMES}
    assert sum(action == "create" for action, _ in state["commands"]) == 2


@pytest.mark.parametrize("closed", ["CLOSED", "MERGED"])
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("has_close_event", [False, True])
def test_any_close_removes_sessions(manager, state, closed, draft, has_close_event):
    populate(manager, state)
    state["pr"].update(state=closed, isDraft=draft)
    state["pr"]["timelineItems"]["nodes"] = [{"id": "CLOSE_1"}] if has_close_event else []
    assert manager.manage_session("owner/repo", "12", "42", "cleanup") == "removed"
    assert not state["volumes"]
    assert manager.manage_session("owner/repo", "12", "42", "review", HEAD) == "skipped"
    assert not state["volumes"]


def test_missing_volumes_are_an_idempotent_cleanup(manager, state):
    state["pr"]["state"] = "CLOSED"
    for _ in range(2):
        assert manager.manage_session("owner/repo", "12", "42", "cleanup") == "removed"
    assert all(action == "inspect" for action, _ in state["commands"])


@pytest.mark.parametrize("review_first", [False, True])
def test_late_cleanup_after_reopen_preserves_fresh_generation(manager, state, review_first):
    populate(manager, state)
    state["pr"]["timelineItems"]["nodes"] = [{"id": "CLOSE_1"}]
    if review_first:
        assert manager.manage_session("owner/repo", "12", "42", "review", HEAD) == "ready"
    expected = "retained" if review_first else "removed"
    assert manager.manage_session("owner/repo", "12", "42", "cleanup") == expected
    assert manager.manage_session("owner/repo", "12", "42", "review", HEAD) == "ready"
    assert state["volumes"] == {name: {manager.EPOCH_LABEL: "closed:CLOSE_1"} for name in NAMES}


def test_close_reopen_close_advances_generation(manager, state):
    populate(manager, state)
    for event in ("CLOSE_1", "CLOSE_2"):
        state["pr"]["state"] = "CLOSED"
        state["pr"]["timelineItems"]["nodes"] = [{"id": event}]
        assert manager.manage_session("owner/repo", "12", "42", "cleanup") == "removed"
        state["pr"]["state"] = "OPEN"
        assert manager.manage_session("owner/repo", "12", "42", "review", HEAD) == "ready"
        assert state["volumes"] == {name: {manager.EPOCH_LABEL: f"closed:{event}"} for name in NAMES}


@pytest.mark.parametrize("operation", ["inspect", "rm", "create"])
def test_docker_failures_are_not_missing_or_ready(manager, state, operation):
    populate(manager, state, "closed:old")
    state["fail"] = operation
    with pytest.raises(subprocess.CalledProcessError):
        manager.manage_session("owner/repo", "12", "42", "review", HEAD)


@pytest.mark.parametrize("kind", ["graphql", "state", "head", "draft", "too_many", "null_event"])
def test_invalid_api_state_never_mutates_volumes(manager, state, monkeypatch, kind):
    if kind == "graphql":
        monkeypatch.setattr(manager, "github_api", lambda *args: {"errors": [{"message": "unavailable"}]})
    elif kind == "state":
        state["pr"]["state"] = "UNKNOWN"
    elif kind == "head":
        state["pr"]["headRefOid"] = "bad"
    elif kind == "draft":
        state["pr"]["isDraft"] = "false"
    elif kind == "too_many":
        state["pr"]["timelineItems"]["nodes"] = [{"id": "one"}, {"id": "two"}]
    else:
        state["pr"]["timelineItems"]["nodes"] = [{"id": None}]
    with pytest.raises(ValueError):
        manager.manage_session("owner/repo", "12", "42", "review", HEAD)
    assert not state["commands"]


@pytest.mark.parametrize(
    "number,repository_id,mode,head",
    [
        ("../12", "42", "cleanup", None),
        ("12", "bad", "cleanup", None),
        ("12", "42", "other", None),
        ("12", "42", "review", None),
    ],
)
def test_invalid_identity_cannot_target_volumes(manager, state, number, repository_id, mode, head):
    with pytest.raises(ValueError):
        manager.manage_session("owner/repo", number, repository_id, mode, head)
    assert not state["commands"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
