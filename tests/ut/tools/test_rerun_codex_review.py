# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify comment routing without issuing GitHub mutations."""

import importlib.util
import io
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


@pytest.fixture
def router(monkeypatch):
    scripts = Path(__file__).resolve().parents[3] / ".github/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("rerun_codex_review", scripts / "rerun_codex_review.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("CODEX_REVIEW_RERUN_TOKEN", "test-only-token")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(
        module.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected subprocess")
    )
    return module


@pytest.fixture
def pr():
    return {
        "number": 2911,
        "state": "open",
        "draft": False,
        "head": {"sha": "a" * 40, "ref": "feature", "repo": {"id": 12}},
        "base": {"ref": "main"},
    }


@pytest.fixture
def run(pr):
    return {
        "id": 123,
        "event": "pull_request_target",
        "path": ".github/workflows/codex-review.yml",
        "head_sha": pr["head"]["sha"],
        "head_branch": "feature",
        "head_repository": {"id": 12},
        "repository": {"full_name": "owner/repo"},
        "pull_requests": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
    }


@pytest.fixture
def api(router, pr, run, monkeypatch):
    calls = []

    def request(endpoint):
        calls.append(endpoint)
        if "/workflows/" in endpoint:
            return {"total_count": 1, "workflow_runs": [run]}
        if "/pulls/" in endpoint:
            return pr
        assert endpoint == "repos/owner/repo/actions/runs/123"
        return run

    monkeypatch.setattr(router, "github_api", request)
    monkeypatch.setattr(router, "snapshot_matches", lambda *args: True)
    return calls


def test_rerun_all_jobs_uses_dedicated_token_and_empty_response(router, pr, api, monkeypatch, tmp_path):
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="")  # GitHub returns empty 201, not JSON.

    monkeypatch.setattr(router.subprocess, "run", execute)
    summary = tmp_path / "summary"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert router.route_review("owner/repo", pr)
    assert len(calls) == 1
    assert calls[0][0] == ["gh", "api", "--method", "POST", "repos/owner/repo/actions/runs/123/rerun"]
    assert calls[0][1]["env"]["GH_TOKEN"] == "test-only-token"
    assert calls[0][1]["check"]
    assert "existing review run" in summary.read_text()


@pytest.mark.parametrize("status", ["queued", "in_progress", "waiting", "pending", "requested"])
def test_active_run_is_reused_without_rerun_token(router, pr, run, api, monkeypatch, status):
    run["status"] = status
    monkeypatch.delenv("CODEX_REVIEW_RERUN_TOKEN")
    assert router.route_review("owner/repo", pr)


@pytest.mark.parametrize(
    "field,value",
    [("state", "closed"), ("draft", True), ("head", {"sha": "c" * 40}), ("base", {"ref": "release"})],
)
def test_pr_change_before_rerun_aborts(router, pr, api, monkeypatch, field, value):
    current = {**pr, field: value}
    original = router.github_api
    monkeypatch.setattr(
        router, "github_api", lambda endpoint: current if "/pulls/" in endpoint else original(endpoint)
    )
    assert router.route_review("owner/repo", pr)


@pytest.mark.parametrize(
    "field,value",
    [
        ("event", "issue_comment"),
        ("path", ".github/workflows/ci.yml"),
        ("repository", {"full_name": "other/repo"}),
        ("head_sha", "c" * 40),
        ("head_branch", "different"),
        ("head_repository", {"id": 13}),
    ],
)
def test_wrong_run_never_reads_artifact(router, run, pr, monkeypatch, field, value):
    run[field] = value
    monkeypatch.setattr(router, "snapshot_matches", lambda *args: pytest.fail("Foreign run"))
    assert not router.matching_run("owner/repo", run, pr)


def test_missing_credential_is_an_error_not_silent_fallback(router, pr, api, monkeypatch):
    monkeypatch.delenv("CODEX_REVIEW_RERUN_TOKEN")
    with pytest.raises(ValueError, match="CODEX_REVIEW_RERUN_TOKEN"):
        router.route_review("owner/repo", pr)


def test_rerun_api_failure_propagates(router, pr, api, monkeypatch):
    def execute(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "gh", stderr="HTTP 403")

    monkeypatch.setattr(router.subprocess, "run", execute)
    with pytest.raises(subprocess.CalledProcessError):
        router.route_review("owner/repo", pr)


def test_expired_run_falls_back(router, pr, run, api):
    run["created_at"] = "2020-01-01T00:00:00Z"
    assert not router.route_review("owner/repo", pr)


def test_unverifiable_run_falls_back(router, pr, api, monkeypatch):
    monkeypatch.setattr(router, "snapshot_matches", lambda *args: False)
    assert not router.route_review("owner/repo", pr)


def test_paginated_search_prefers_newest_matching_run(router, pr, run, monkeypatch):
    calls = []

    def request(endpoint):
        calls.append(endpoint)
        if endpoint.endswith("&page=1"):
            return {"total_count": 2, "workflow_runs": [{**run, "id": 100, "head_sha": "c" * 40}]}
        if endpoint.endswith("&page=2"):
            return {"total_count": 2, "workflow_runs": [run]}
        return pr if "/pulls/" in endpoint else {**run, "status": "in_progress"}

    monkeypatch.setattr(router, "github_api", request)
    monkeypatch.setattr(router, "snapshot_matches", lambda *args: True)
    assert router.route_review("owner/repo", pr)
    assert len(calls) == 4
    assert "page=2" in calls[1]


def install_snapshot(router, monkeypatch, pr, snapshot=None, name="discussion.json"):
    data = {"repository": "owner/repo", "number": 2911, "head": pr["head"]["sha"], "base_ref": "main"}
    if snapshot:
        data.update(snapshot)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, json.dumps(data))
    monkeypatch.setattr(
        router,
        "github_api",
        lambda endpoint: {
            "artifacts": [
                {
                    "id": 9,
                    "name": "codex-context-123-1",
                    "expired": False,
                    "size_in_bytes": len(buffer.getvalue()),
                }
            ]
        },
    )

    def execute(command, **kwargs):
        assert command == ["gh", "api", "repos/owner/repo/actions/artifacts/9/zip"]
        return SimpleNamespace(stdout=buffer.getvalue())

    monkeypatch.setattr(router.subprocess, "run", execute)


def test_fork_run_without_pr_links_verified_from_snapshot(router, run, pr, monkeypatch):
    install_snapshot(router, monkeypatch, pr)
    assert run["pull_requests"] == []
    assert router.matching_run("owner/repo", run, pr)


@pytest.mark.parametrize(
    "snapshot", [{"number": 2912}, {"head": "c" * 40}, {"base_ref": "release"}, {"repository": "other/repo"}]
)
def test_snapshot_rejects_wrong_pr_head_or_base(router, run, pr, monkeypatch, snapshot):
    install_snapshot(router, monkeypatch, pr, snapshot)
    assert not router.matching_run("owner/repo", run, pr)


def test_artifact_cannot_extract_arbitrary_paths(router, run, pr, monkeypatch):
    install_snapshot(router, monkeypatch, pr, name="../discussion.json")
    with pytest.raises(ValueError, match="only discussion.json"):
        router.snapshot_matches("owner/repo", run, pr)


def test_no_snapshot_falls_back_without_download(router, run, pr, monkeypatch):
    monkeypatch.setattr(router, "github_api", lambda endpoint: {"artifacts": []})
    assert not router.snapshot_matches("owner/repo", run, pr)


def test_target_manifest_survives_expired_discussion(router, run, pr, monkeypatch):
    install_snapshot(router, monkeypatch, pr, name="review-target.json")
    monkeypatch.setattr(
        router,
        "github_api",
        lambda endpoint: {
            "artifacts": [
                {"id": 8, "name": "codex-context-123-1", "expired": True, "size_in_bytes": 100},
                {"id": 9, "name": "codex-target-123-1", "expired": False, "size_in_bytes": 100},
            ]
        },
    )
    assert router.matching_run("owner/repo", run, pr)


def test_latest_attempt_identity_wins(router, run, pr, monkeypatch):
    install_snapshot(router, monkeypatch, pr, {"base_ref": "other"}, name="review-target.json")
    monkeypatch.setattr(
        router,
        "github_api",
        lambda endpoint: {
            "artifacts": [
                {"id": 7, "name": "codex-context-123-1", "expired": False, "size_in_bytes": 100},
                {"id": 9, "name": "codex-target-123-2", "expired": False, "size_in_bytes": 100},
            ]
        },
    )
    assert not router.matching_run("owner/repo", run, pr)


def test_oversized_snapshot_is_rejected_before_download(router, run, pr, monkeypatch):
    monkeypatch.setattr(
        router,
        "github_api",
        lambda endpoint: {
            "artifacts": [
                {
                    "id": 9,
                    "name": "codex-target-123-1",
                    "expired": False,
                    "size_in_bytes": router.MAX_CONTEXT_BYTES + 1,
                },
            ]
        },
    )
    with pytest.raises(ValueError, match="size limit"):
        router.snapshot_matches("owner/repo", run, pr)


def test_workflow_limits_token_to_trusted_step():
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/codex-review.yml").read_text())
    jobs = workflow["jobs"]
    prepare = jobs["prepare"]
    assert prepare["permissions"]["actions"] == "read"
    assert prepare["concurrency"]["cancel-in-progress"] is False
    contexts = [step for step in prepare["steps"] if step.get("id") == "context"]
    assert len(contexts) == 1
    assert "issue_comment" in contexts[0]["env"]["CODEX_REVIEW_RERUN_TOKEN"]
    assert "secrets.CODEX_REVIEW_RERUN_TOKEN" in contexts[0]["env"]["CODEX_REVIEW_RERUN_TOKEN"]
    for name, job in jobs.items():
        if name != "prepare":
            assert "CODEX_REVIEW_RERUN_TOKEN" not in json.dumps(job)
    identity = next(step for step in prepare["steps"] if step["name"] == "Upload review routing identity")
    discussion = next(step for step in prepare["steps"] if step["name"] == "Upload discussion snapshot")
    assert identity["with"]["retention-days"] == 30
    assert discussion["with"]["retention-days"] == 1
    ci = yaml.safe_load((root / ".github/workflows/ci.yml").read_text())
    assert "test_rerun_codex_review.py" in json.dumps(ci["jobs"]["ci-helper-tests"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
