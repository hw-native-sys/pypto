# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise CI routing and merge-gate failures without GitHub writes or devices."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
REPO = "owner/repo"
HEAD, BASE, MERGE = "a" * 40, "b" * 40, "c" * 40


@pytest.fixture
def controller(monkeypatch):
    spec = importlib.util.spec_from_file_location("ci_control", ROOT / ".github/scripts/ci_control.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("CI_ROUTED_EVENT", raising=False)
    monkeypatch.setenv("CI_ROUTED_HEAD", HEAD)
    monkeypatch.setenv("GITHUB_RUN_ID", "900")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected subprocess"))
    return module


@pytest.fixture
def control(controller, monkeypatch):
    # Most cases isolate other API contracts; live-ref tests use controller.
    monkeypatch.setattr(controller, "current_base", lambda repo: BASE)
    monkeypatch.setattr(controller, "verify_branch_freshness", lambda *args: None)
    return controller


@pytest.fixture
def pr():
    return {
        "state": "open",
        "number": 123,
        "created_at": "2026-01-01T00:00:00Z",
        "mergeable": True,
        "merge_commit_sha": MERGE,
        "head": {"sha": HEAD, "repo": {"id": 2}},
        "base": {"sha": BASE, "ref": "main", "repo": {"id": 1, "full_name": REPO}},
    }


@pytest.fixture
def event():
    return {
        "action": "created",
        "issue": {"number": 123, "pull_request": {}},
        "comment": {"body": "/ci full", "user": {"login": "external-user"}, "author_association": "NONE"},
    }


@pytest.fixture
def merge_clock(control, monkeypatch):
    elapsed = [0.0]
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        elapsed[0] += seconds

    monkeypatch.setattr(control, "time", SimpleNamespace(monotonic=lambda: elapsed[0], sleep=sleep))
    return waits


@pytest.fixture
def needs(control):
    result: dict[str, Any] = {name: {"result": "success"} for name in control.REQUIRED | control.OPTIONAL}
    result["changes"]["outputs"] = {"docs_changed": "false", "ptoas_version_changed": "false"}
    result.update({name: {"result": "skipped"} for name in control.CONDITIONAL})
    return result


@pytest.mark.parametrize("job", ["system-tests", "onboard-tests-a5"])
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_required_core_results_block_merge(control, needs, job, result):
    # A skipped required job must still block the aggregate merge check.
    assert job in control.REQUIRED
    needs[job]["result"] = result
    with pytest.raises(ValueError, match=job):
        control.aggregate(needs)


@pytest.mark.parametrize("job", ["prebuild-wheels", "prebuild-wheels-github"])
def test_warmup_failure_does_not_hide_successful_tests(control, needs, job):
    needs[job]["result"] = "failure"
    control.aggregate(needs)


@pytest.mark.parametrize(
    "job,flag",
    [
        ("docs-examples", "docs_changed"),
        ("serving-guard-a2a3", "ptoas_version_changed"),
        ("serving-guard-a5", "ptoas_version_changed"),
    ],
)
@pytest.mark.parametrize("result", ["skipped", "failure", "cancelled"])
def test_applicable_conditional_job_must_run(control, needs, job, flag, result):
    for name in control.CONDITIONAL:
        needs[name]["result"] = "success"
    for guard, output in control.SERVING_RESULTS.items():
        needs[guard]["outputs"] = {output: "success"}
    needs[job]["result"] = result
    needs["changes"]["outputs"][flag] = "true"
    with pytest.raises(ValueError, match=job):
        control.aggregate(needs)
    needs[job]["result"] = "success"
    control.aggregate(needs)


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
@pytest.mark.parametrize("actual", ["skipped", "failure", "cancelled", None])
def test_successful_call_requires_its_selected_serving_job(control, needs, platform, actual):
    needs["changes"]["outputs"]["ptoas_version_changed"] = "true"
    for name, output in control.SERVING_RESULTS.items():
        needs[name] = {"result": "success", "outputs": {output: "success"}}
    name = f"serving-guard-{platform}"
    needs[name]["outputs"] = {f"{platform}-result": actual} if actual else {}
    with pytest.raises(ValueError, match=name):
        control.aggregate(needs)


def test_unknown_classification_blocks_merge(control, needs):
    del needs["changes"]["outputs"]["docs_changed"]
    with pytest.raises(ValueError, match="docs-examples"):
        control.aggregate(needs)


def test_missing_dependency_cannot_silently_reduce_coverage(control, needs):
    del needs["unit-tests"]
    with pytest.raises(ValueError, match="dependency set"):
        control.aggregate(needs)


def test_failed_matrix_blocks_merge(control, needs):
    needs["pypto-lib-compile"]["result"] = "failure"
    with pytest.raises(ValueError, match="pypto-lib-compile"):
        control.aggregate(needs)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("pull_request", "false"),
        ("push", "true"),
        ("workflow_dispatch", "true"),
    ],
)
@pytest.mark.parametrize("enabled", [None, "", "false", "true"])
@pytest.mark.parametrize("fork", [False, True])
def test_event_selects_tier_without_repository_variables(
    control, monkeypatch, tmp_path, pr, kind, expected, enabled, fork
):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    monkeypatch.setenv("GITHUB_EVENT_NAME", kind)
    monkeypatch.setenv("GITHUB_SHA", MERGE)
    # Fork runs supplied an empty value even with the base repository switch
    # enabled. Missing or stale rollout settings must never opt a PR into full.
    if enabled is None:
        monkeypatch.delenv("CI_COMMENT_TIERS_ENABLED", raising=False)
    else:
        monkeypatch.setenv("CI_COMMENT_TIERS_ENABLED", enabled)
    if not fork:
        pr["head"]["repo"] = pr["base"]["repo"]
    control.prepare(REPO, {"pull_request": pr} if kind == "pull_request" else {})
    assert target.read_text() == f"full={expected}\ncheckout_sha={MERGE}\nreuse_lint=false\n"


@pytest.mark.parametrize("parents", [[BASE, HEAD], [HEAD, BASE], [BASE, "d" * 40]])
def test_merge_revision_is_verified(control, monkeypatch, pr, parents):
    monkeypatch.setattr(control, "api", lambda path: {"parents": [{"sha": value} for value in parents]})
    if parents == [BASE, HEAD]:
        control.verify_merge(REPO, pr, HEAD, BASE, MERGE)
    else:
        with pytest.raises(ValueError, match="parents"):
            control.verify_merge(REPO, pr, HEAD, BASE, MERGE)


@pytest.mark.parametrize("invalid", ["", "ref", "type", "sha"])
def test_current_base_requires_live_main_commit(controller, monkeypatch, invalid):
    ref: dict[str, Any] = {"ref": "refs/heads/main", "object": {"type": "commit", "sha": BASE}}
    if invalid == "ref":
        ref["ref"] = "refs/heads/feature"
    elif invalid == "type":
        ref["object"]["type"] = "tag"
    elif invalid == "sha":
        ref["object"]["sha"] = "not-a-sha"

    def api(path):
        assert path == f"repos/{REPO}/git/ref/heads/main"
        return ref

    monkeypatch.setattr(controller, "api", api)
    if invalid:
        with pytest.raises(ValueError, match="Expected"):
            controller.current_base(REPO)
    else:
        assert controller.current_base(REPO) == BASE


@pytest.mark.parametrize("behind", [0, 4, 5, 9, 10, 11, 12, 1000])
def test_branch_freshness_counts_only_missing_main_commits(controller, monkeypatch, behind):
    def api(path):
        assert path == f"repos/{REPO}/compare/{BASE}...{HEAD}?per_page=1"
        # A long PR history is allowed; neither ahead_by nor a paginated commit
        # list measures how far the PR is behind main.
        return {"behind_by": behind, "ahead_by": 200, "commits": [{}]}

    monkeypatch.setattr(controller, "api", api)
    if behind <= 10:
        controller.verify_branch_freshness(REPO, HEAD, BASE)
    else:
        with pytest.raises(ValueError, match=f"{behind} commits behind main.*rebase"):
            controller.verify_branch_freshness(REPO, HEAD, BASE)


@pytest.mark.parametrize("behind", [None, -1, True, "4", 4.5])
def test_unknown_branch_distance_cannot_authorize_full(controller, monkeypatch, behind):
    monkeypatch.setattr(controller, "api", lambda path: {"behind_by": behind})
    with pytest.raises(ValueError, match="Cannot determine"):
        controller.verify_branch_freshness(REPO, HEAD, BASE)


@pytest.mark.parametrize("command", ["command-route", "dispatch"])
@pytest.mark.parametrize("behind", [11, 12])
def test_outdated_branch_blocks_full_and_reports_rebase(
    controller, monkeypatch, event, pr, tmp_path, command, behind
):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event))
    output_file = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("GITHUB_RUN_ID", "50")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setattr(controller.sys, "argv", ["ci_control.py", command])

    def api(path):
        if path.endswith("/pulls/123"):
            return pr
        if path.endswith("/git/ref/heads/main"):
            return {"ref": "refs/heads/main", "object": {"type": "commit", "sha": BASE}}
        assert path == f"repos/{REPO}/compare/{BASE}...{HEAD}?per_page=1"
        return {"behind_by": behind}

    monkeypatch.setattr(controller, "api", api)
    # Only GET metadata is accepted above: rejection precedes pending status
    # writes, device dispatch, and the merge-computation polling loop.
    with pytest.raises(SystemExit) as error:
        controller.main()
    assert error.value.code == 1
    key, reason = output_file.read_text().strip().split("=", 1)
    assert key == "command_error"
    monkeypatch.setenv("CI_COMMAND_ERROR", reason)
    monkeypatch.setattr(controller, "pages", lambda path: [])
    replies = []
    monkeypatch.setattr(controller, "api", lambda path, payload: replies.append((path, payload)))
    controller.command_feedback(REPO, event)
    assert len(replies) == 1
    assert replies[0][0] == f"repos/{REPO}/issues/123/comments"
    assert f"{behind} commits behind main" in replies[0][1]["body"]
    assert "rebase onto the latest main, wait for lint to pass" in replies[0][1]["body"]


@pytest.mark.parametrize("behind", [0, 4, 5, 9, 10, 11, 12])
def test_manual_pr_full_cannot_bypass_branch_distance(controller, monkeypatch, pr, tmp_path, behind):
    output_file = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_SHA", BASE)
    monkeypatch.setattr(controller, "pull", lambda *args: pr)
    monkeypatch.setattr(controller, "verify_merge", lambda *args: BASE)
    monkeypatch.setattr(controller, "current_base", lambda repo: BASE)
    lint_checks = []
    monkeypatch.setattr(controller, "matching_lint_run", lambda *args: lint_checks.append(True))

    def api(path):
        assert path == f"repos/{REPO}/compare/{BASE}...{HEAD}?per_page=1"
        return {"behind_by": behind}

    monkeypatch.setattr(controller, "api", api)
    event = {"inputs": {"pr_number": "123", "head_sha": HEAD, "base_sha": BASE, "merge_sha": MERGE}}
    if behind <= 10:
        controller.prepare(REPO, event)
        assert f"checkout_sha={MERGE}\nreuse_lint=true\n" in output_file.read_text()
        assert lint_checks == [True]
    else:
        with pytest.raises(ValueError, match="rebase"):
            controller.prepare(REPO, event)
        assert not output_file.exists() and not lint_checks


@pytest.mark.parametrize("snapshot_behind", [0, 1, 4, 5, 9, 10])
def test_stale_pr_base_allows_pinned_full_lifecycle(
    controller, monkeypatch, event, pr, tmp_path, snapshot_behind
):
    # PR metadata and the actual merge may both lag live main, as in #3400.
    # Keep the actual merge parents rather than relabelling the tested base.
    stale = "d" * 40
    tested = stale if snapshot_behind else BASE
    pr["base"]["sha"] = stale
    output_file = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_SHA", BASE)
    monkeypatch.setattr(controller.time, "sleep", lambda *args: pytest.fail("No merge wait is needed"))
    run = {
        "id": 50,
        "run_attempt": 1,
        "run_number": 8,
        "display_title": f"CI full PR #123 @{HEAD} base:{tested}",
        "created_at": "2026-11-05T01:00:00Z",
        "event": "workflow_dispatch",
        "path": controller.WORKFLOW,
        "workflow_id": 99,
        "head_branch": "main",
        "repository": {"full_name": REPO, "id": 1},
        "head_repository": {"id": 1},
        "status": "completed",
        "conclusion": "success",
    }
    lint = {
        **run,
        "id": 40,
        "display_title": f"CI lint PR #123 @{HEAD} base:{stale}",
        "event": "pull_request",
        "path": controller.LINT_WORKFLOW,
        "workflow_id": 98,
        "head_sha": HEAD,
        "head_repository": {"id": 2},
    }
    writes = []
    statuses = []
    responses = {
        f"repos/{REPO}/pulls/123": pr,
        f"repos/{REPO}/git/ref/heads/main": {
            "ref": "refs/heads/main",
            "object": {"type": "commit", "sha": BASE},
        },
        f"repos/{REPO}/compare/{BASE}...{HEAD}": {"behind_by": 4, "ahead_by": 20},
        f"repos/{REPO}/compare/{stale}...{BASE}": {
            "merge_base_commit": {"sha": stale},
            "ahead_by": snapshot_behind,
        },
        f"repos/{REPO}/git/commits/{MERGE}": {"parents": [{"sha": tested}, {"sha": HEAD}]},
        f"repos/{REPO}/actions/workflows/ci.yml": {"id": 98},
        f"repos/{REPO}/actions/workflows/ci-full.yml": {"id": 99},
        f"repos/{REPO}/actions/workflows/ci.yml/runs": {"workflow_runs": [lint]},
        f"repos/{REPO}/actions/runs": {"workflow_runs": []},
        f"repos/{REPO}/commits/{HEAD}/status": {"statuses": statuses},
        f"repos/{REPO}/actions/runs/50": run,
        f"repos/{REPO}/actions/runs/40/attempts/1/jobs": {
            "jobs": [
                {"name": name, "status": "completed", "conclusion": "success"}
                for name in ("changes", "pre-commit", "clang-tidy")
            ]
        },
        f"repos/{REPO}/actions/runs/50/attempts/1/jobs": {
            "jobs": [{"name": "full-result", "conclusion": "success"}]
        },
    }

    def api(path, payload=None):
        if payload is not None:
            writes.append((path, payload))
            if path.endswith(f"/statuses/{HEAD}"):
                statuses[:] = [payload]
            elif path.endswith("/dispatches"):
                return {"workflow_run_id": 50}
            else:
                pytest.fail(f"Unexpected write: {path}")
            return None
        endpoint = path.split("?", 1)[0]
        assert endpoint in responses, f"Unexpected read: {path}"
        return responses[endpoint]

    monkeypatch.setattr(controller, "api", api)
    controller.command_route(REPO, event)
    controller.dispatch(REPO, event)
    inputs = writes[1][1]["inputs"]
    assert inputs == {
        "pr_number": "123",
        "head_sha": HEAD,
        "base_sha": tested,
        "merge_sha": MERGE,
        "request_id": "900-1",
    }
    controller.prepare(REPO, {"inputs": inputs})
    assert f"checkout_sha={MERGE}\nreuse_lint=true\n" in output_file.read_text()
    controller.publish(REPO, {"workflow_run": {"id": 50}})
    assert statuses[0]["state"] == "success"
    assert statuses[0]["target_url"].endswith("/runs/50/attempts/1")
    # A repeat comment must recognize this run despite the old PR base field.
    assert controller.already_running_or_passed(REPO, pr)
    assert pr["base"]["sha"] == stale, "Do not rewrite the lint snapshot"


def test_changed_pr_rejected_before_checkout(control, pr):
    pr["head"]["sha"] = "d" * 40
    with pytest.raises(ValueError, match="PR changed"):
        control.verify_merge(REPO, pr, HEAD, BASE, MERGE)


@pytest.mark.parametrize("behind", [None, -1, True, "4", 4.5, 11, 15])
@pytest.mark.parametrize("stage", ["admission", "prepare", "result"])
def test_old_or_unknown_snapshot_cannot_pass_with_a_fresh_head(
    controller, monkeypatch, pr, event, tmp_path, behind, stage
):
    live = "d" * 40
    monkeypatch.setattr(controller, "current_base", lambda repo: live)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_SHA", live)
    output_file = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setattr(controller.time, "sleep", lambda *a: pytest.fail("Reject without polling"))

    def api(path, payload=None):
        assert payload is None, "Snapshot rejection must precede status writes and dispatch"
        if path.endswith("/pulls/123"):
            return pr
        if path.endswith(f"/git/commits/{MERGE}"):
            return {"parents": [{"sha": BASE}, {"sha": HEAD}]}
        if path == f"repos/{REPO}/compare/{live}...{HEAD}?per_page=1":
            return {"behind_by": 0}
        assert path == f"repos/{REPO}/compare/{BASE}...{live}?per_page=1"
        return {"merge_base_commit": {"sha": BASE}, "ahead_by": behind}

    monkeypatch.setattr(controller, "api", api)
    with pytest.raises(ValueError, match="tested base|Tested base"):
        if stage == "admission":
            controller.dispatch(REPO, event)
        elif stage == "prepare":
            controller.prepare(
                REPO, {"inputs": {"pr_number": "123", "head_sha": HEAD, "base_sha": BASE, "merge_sha": MERGE}}
            )
        else:
            controller.verify_result_freshness(REPO, HEAD, BASE)
    assert not output_file.exists()


@pytest.mark.parametrize("invalid", ["rewritten", "wrong-head", "wrong-base", "conflict", "unknown"])
def test_snapshot_reuse_keeps_revision_and_conflict_guards(control, monkeypatch, pr, invalid):
    live = "d" * 40
    monkeypatch.setattr(control, "current_base", lambda repo: live)
    if invalid in {"conflict", "unknown"}:
        pr["mergeable"] = False if invalid == "conflict" else None

    def api(path):
        if path.endswith(f"/git/commits/{MERGE}"):
            return {"parents": [{"sha": BASE}, {"sha": "e" * 40 if invalid == "wrong-head" else HEAD}]}
        assert path == f"repos/{REPO}/compare/{BASE}...{live}?per_page=1"
        return {"merge_base_commit": {"sha": "e" * 40}, "ahead_by": 1}

    monkeypatch.setattr(control, "api", api)
    pinned = live if invalid == "wrong-base" else None
    with pytest.raises(ValueError):
        control.verify_merge(REPO, pr, HEAD, pinned, MERGE)


@pytest.mark.parametrize("failure", ["", "pending", "dispatch", "bind", "ack", "observer"])
@pytest.mark.parametrize("association", ["NONE", "CONTRIBUTOR", "MEMBER", "OWNER"])
def test_any_commenter_can_dispatch_pinned_fork_pr(control, monkeypatch, event, pr, failure, association):
    event["comment"]["author_association"] = association
    monkeypatch.setattr(control, "matching_lint_run", lambda *args: {"id": 40})
    monkeypatch.setattr(control, "already_running_or_passed", lambda *args: False)
    writes = []
    gate = {"state": "success"}

    def api(path, payload=None):
        if payload is not None:
            if path.endswith(f"/statuses/{HEAD}"):
                if failure == "pending" or (failure == "bind" and len(writes) == 2):
                    raise subprocess.CalledProcessError(1, ["gh", "api"])
                gate["state"] = payload["state"]
            else:
                assert gate["state"] == "pending", "Dispatch must not leave the old success visible"
                if failure == "dispatch":
                    raise subprocess.CalledProcessError(1, ["gh", "api"])
                if failure == "observer" and "ci-full-progress.yml" in path:
                    raise subprocess.CalledProcessError(1, ["gh", "api"])
            writes.append((path, payload))
            if path.endswith("/dispatches"):
                return None if failure == "ack" else {"workflow_run_id": 51}
        elif "/actions/runs?" in path:
            return {"workflow_runs": []}
        elif "/pulls/" in path:
            return pr
        elif "/git/commits/" in path:
            return {"parents": [{"sha": BASE}, {"sha": HEAD}]}
        else:
            raise AssertionError(path)

    monkeypatch.setattr(control, "api", api)
    # Empty pull_request data is still a PR marker; the issue API link normally
    # has fields, but identity should depend on its presence rather than contents.
    event["issue"]["pull_request"] = {"url": "unused"}
    if failure:
        error = ValueError if failure == "ack" else subprocess.CalledProcessError
        with pytest.raises(error):
            control.dispatch(REPO, event)
        assert gate["state"] == ("success" if failure == "pending" else "pending")
        assert len(writes) == {"pending": 0, "dispatch": 1, "bind": 2, "ack": 2, "observer": 3}[failure]
        if failure in {"pending", "dispatch"}:
            assert all("/dispatches" not in path for path, _ in writes)
        return
    control.dispatch(REPO, event)
    assert writes == [
        (
            f"repos/{REPO}/statuses/{HEAD}",
            {
                "context": "ci/full",
                "state": "pending",
                "description": "Full CI requested; waiting for dispatch (request 900-1)",
                "target_url": f"https://github.com/{REPO}/pull/123",
            },
        ),
        (
            f"repos/{REPO}/actions/workflows/ci-full.yml/dispatches",
            {
                "ref": "main",
                "return_run_details": True,
                "inputs": {
                    "pr_number": "123",
                    "head_sha": HEAD,
                    "base_sha": BASE,
                    "merge_sha": MERGE,
                    "request_id": "900-1",
                },
            },
        ),
        (
            f"repos/{REPO}/statuses/{HEAD}",
            {
                "context": "ci/full",
                "state": "pending",
                "description": "Full CI dispatched; waiting for results",
                "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1",
            },
        ),
        (
            f"repos/{REPO}/actions/workflows/ci-full-progress.yml/dispatches",
            {"ref": "main", "inputs": {"run_id": "51"}},
        ),
    ]


@pytest.mark.parametrize("association", ["NONE", "CONTRIBUTOR", "MEMBER", "OWNER"])
@pytest.mark.parametrize("body", ["/ci full", "/ci full extra"])
def test_command_queue_admits_any_commenter(control, monkeypatch, event, pr, tmp_path, association, body):
    output_file = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    event["comment"]["body"] = body
    event["comment"]["author_association"] = association

    def request(path):
        if "/pulls/" in path:
            return pr
        assert path.endswith(f"/git/commits/{MERGE}")
        return {"parents": [{"sha": BASE}, {"sha": HEAD}]}

    monkeypatch.setattr(control, "api", request)
    control.command_route(REPO, event)
    if body == "/ci full":
        assert output_file.read_text() == f"key=123-{HEAD}\nhead={HEAD}\n"
    else:
        assert not output_file.exists()


@pytest.mark.parametrize("routed_head", ["", "d" * 40])
def test_dispatch_cannot_write_outside_admitted_head(control, monkeypatch, event, pr, routed_head):
    monkeypatch.setenv("CI_ROUTED_HEAD", routed_head)
    monkeypatch.setattr(control, "validated_request", lambda *args: pr)
    monkeypatch.setattr(control, "api", lambda *args: pytest.fail("No writes outside the admitted queue"))
    with pytest.raises(ValueError, match="head changed after queue admission"):
        control.dispatch(REPO, event)


@pytest.mark.parametrize("command", ["command_route", "dispatch"])
@pytest.mark.parametrize("invalid", ["closed", "other-branch", "other-repo", "deleted-fork"])
def test_any_commenter_still_requires_valid_pr(control, monkeypatch, event, pr, command, invalid):
    # Opening comment access must not broaden which PRs the trusted workflow runs.
    if invalid == "closed":
        pr["state"] = "closed"
    elif invalid == "other-branch":
        pr["base"]["ref"] = "release"
    elif invalid == "other-repo":
        pr["base"]["repo"]["full_name"] = "other/repo"
    else:
        pr["head"]["repo"] = None

    def request(path):
        assert path == f"repos/{REPO}/pulls/123"
        return pr

    monkeypatch.setattr(control, "api", request)
    with pytest.raises(ValueError, match="open PR targeting this repository's main branch"):
        getattr(control, command)(REPO, event)


@pytest.mark.parametrize("command", ["command_route", "dispatch"])
@pytest.mark.parametrize("invalid", ["edited", "deleted", "issue"])
def test_only_new_pr_comments_are_accepted(control, monkeypatch, event, command, invalid):
    if invalid == "issue":
        del event["issue"]["pull_request"]
    else:
        event["action"] = invalid
    monkeypatch.setattr(control, "api", lambda *a: pytest.fail("Unexpected API call"))
    getattr(control, command)(REPO, event)


@pytest.mark.parametrize("body", ["please /ci full", "/ci full extra", "/ci full\necho injected"])
def test_only_exact_command_is_accepted(control, monkeypatch, event, body):
    event["comment"]["body"] = body
    monkeypatch.setattr(control, "api", lambda *a: pytest.fail("Unexpected API call"))
    control.dispatch(REPO, event)


@pytest.fixture
def publication(control, monkeypatch, pr):
    run = {
        "id": 50,
        "run_attempt": 2,
        "run_number": 8,
        "display_title": f"CI full PR #123 @{HEAD} base:{BASE}",
        "created_at": "2026-11-05T01:00:00Z",
        "event": "workflow_dispatch",
        "path": control.WORKFLOW,
        "workflow_id": 99,
        "head_branch": "main",
        "repository": {"full_name": REPO, "id": 1},
        "head_repository": {"id": 1},
        "status": "completed",
        "conclusion": "success",
    }
    jobs = [{"name": "full-result", "conclusion": "success"}]
    candidates = [run]
    writes = []

    def api(path, payload=None):
        if payload is not None:
            writes.append((path, payload))
        elif path.endswith("/actions/runs/50"):
            return run
        elif path.endswith("/workflows/ci-full.yml"):
            return {"id": 99, "path": control.WORKFLOW}
        elif path.endswith("/pulls/123"):
            return pr
        elif "event=workflow_dispatch" in path:
            assert "created=>=2026-11-05T01:00:00Z" in path
            return {"workflow_runs": candidates}
        else:
            raise AssertionError(path)

    def pages(path, key):
        if key == "statuses":
            assert path == f"repos/{REPO}/commits/{HEAD}/status"
            return [
                {
                    "context": "ci/full",
                    "state": "success",
                    "target_url": f"https://github.com/{REPO}/actions/runs/50/attempts/1",
                }
            ]
        if path.endswith("/attempts/2/jobs"):
            return jobs
        assert "event=workflow_dispatch" in path
        return candidates

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setattr(control, "pages", pages)
    return run, jobs, candidates, writes


def test_native_rerun_updates_original_head(control, publication):
    _, _, _, writes = publication
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert writes[0][0] == f"repos/{REPO}/statuses/{HEAD}"
    assert writes[0][1]["state"] == "success"
    assert writes[0][1]["target_url"].endswith("/runs/50/attempts/2")


@pytest.mark.parametrize("conclusion", ["skipped", "failure", "cancelled", None])
def test_non_success_finalizer_never_marks_green(control, publication, conclusion):
    _, jobs, _, writes = publication
    jobs[0]["conclusion"] = conclusion
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert writes[0][1]["state"] == "failure"


def test_missing_finalizer_in_current_attempt_blocks_merge(control, publication):
    _, jobs, _, writes = publication
    jobs.clear()
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert writes[0][1]["state"] == "failure"


def test_delayed_completion_cannot_green_running_rerun(control, publication):
    run, _, _, writes = publication
    run["status"] = "in_progress"
    control.publish(REPO, {"workflow_run": {"id": 50, "status": "completed"}})
    assert writes[0][1]["state"] == "pending"


@pytest.mark.parametrize("change", ["head", "closed", "base_ref", "base_repo"])
@pytest.mark.parametrize(
    "run_status,conclusion", [("in_progress", None), ("completed", "failure"), ("completed", "success")]
)
def test_obsolete_pr_invalidates_original_head_before_it_can_return(
    control, publication, pr, change, run_status, conclusion
):
    original_pr = copy.deepcopy(pr)
    run, _, _, writes = publication
    run.update(status=run_status, conclusion=conclusion)
    if change == "head":
        pr["head"]["sha"] = "d" * 40
    elif change == "closed":
        pr["state"] = "closed"
    elif change == "base_ref":
        pr["base"]["ref"] = "release"
    else:
        pr["base"]["repo"]["full_name"] = "different/repo"
    control.publish(REPO, {"workflow_run": {"id": 50}})
    # Resetting/reopening the PR after publication must not revive the old green.
    pr.update(original_pr)
    assert len(writes) == 1
    assert writes[0][0] == f"repos/{REPO}/statuses/{HEAD}"
    assert writes[0][1]["state"] == "failure"
    assert writes[0][1]["target_url"].endswith("/runs/50/attempts/2")


@pytest.mark.parametrize("run_status", ["in_progress", "completed"])
@pytest.mark.parametrize("newer_lookup_fails", [False, True])
def test_moved_head_preserves_only_verified_newer_status(
    control, monkeypatch, publication, pr, run_status, newer_lookup_fails
):
    run, _, _, writes = publication
    run.update(status=run_status, conclusion="failure")
    pr["head"]["sha"] = "d" * 40
    candidate = copy.deepcopy(run)
    candidate.update(id=51, run_number=9, status="completed", conclusion="success")
    original_api = control.api

    def request(path, payload=None):
        if path.endswith("/actions/runs/51"):
            if newer_lookup_fails:
                raise subprocess.CalledProcessError(1, ["gh", "api"])
            return candidate
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {
                "context": "ci/full",
                "state": "success",
                "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1",
            }
        ],
    )
    control.publish(REPO, {"workflow_run": {"id": 50}})
    if newer_lookup_fails:
        assert len(writes) == 1
        assert writes[0][0] == f"repos/{REPO}/statuses/{HEAD}"
        assert writes[0][1]["state"] == "failure"
    else:
        assert not writes


def test_old_request_cannot_overwrite_newer_request(control, publication):
    run, _, candidates, writes = publication
    newer = copy.deepcopy(run)
    newer.update(id=51, run_number=9, status="queued")
    candidates.append(newer)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not writes


@pytest.mark.parametrize("branch", ["feature", "release-tag"])
@pytest.mark.parametrize("phase", ["in_progress", "completed"])
def test_feature_dispatch_cannot_suppress_genuine_result(control, publication, branch, phase):
    run, jobs, candidates, writes = publication
    run["status"] = phase
    run["conclusion"] = "failure"
    jobs[0]["conclusion"] = "failure"
    newer = copy.deepcopy(run)
    newer.update(id=51, run_number=9, head_branch=branch)
    candidates.append(newer)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert len(writes) == 1
    assert writes[0][1]["state"] == ("pending" if phase == "in_progress" else "failure")


@pytest.mark.parametrize("branch", ["feature", "release-tag"])
def test_feature_dispatch_cannot_own_fallback_status(control, monkeypatch, publication, branch):
    run, _, _, writes = publication
    newer = copy.deepcopy(run)
    newer.update(id=51, run_number=9, head_branch=branch)
    original_api = control.api

    def request(path, payload=None):
        if path.endswith("/pulls/123"):
            raise ValueError("PR metadata unavailable")
        if path.endswith("/actions/runs/51"):
            return newer
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {"context": "ci/full", "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1"}
        ],
    )
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert len(writes) == 1
    assert writes[0][1]["state"] == "failure"


@pytest.mark.parametrize(
    "field,value", [("head_branch", "feature"), ("workflow_id", 100), ("path", ".github/workflows/fake.yml")]
)
def test_title_alone_cannot_authorize_status(control, publication, field, value):
    publication[0][field] = value
    with pytest.raises(ValueError, match="trusted workflow"):
        control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not publication[3]


def test_workflow_wiring_covers_every_full_job(control):
    workflow = yaml.safe_load((ROOT / control.WORKFLOW).read_text())
    assert isinstance(workflow, dict)
    jobs = workflow["jobs"]
    assert set(jobs["full-result"]["needs"]) == set(jobs) - {"full-result"}
    assert set(jobs["full-result"]["needs"]) == control.REQUIRED | control.OPTIONAL | set(control.CONDITIONAL)
    for name, job in jobs.items():
        if name not in {"changes", "toolchain", "pre-commit", "clang-tidy", "lint-gate", "full-result"}:
            assert "needs.changes.outputs.full == 'true'" in job["if"]
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/checkout@"):
                options = step["with"]
                expected = (
                    "github.workflow_sha"
                    if "sparse-checkout" in options
                    else "needs.changes.outputs.checkout_sha"
                )
                if name == "changes":
                    expected = (
                        "github.event_name == 'pull_request' && "
                        "github.event.pull_request.base.sha || github.workflow_sha"
                    )
                assert options["ref"] == "${{ " + expected + " }}"
                assert options["persist-credentials"] is False
            if step.get("uses") == "./.github/actions/setup-ci-job":
                assert step["with"]["checkout-ref"] == "${{ needs.changes.outputs.checkout_sha }}"
    for platform in ("a2a3", "a5"):
        guard = jobs[f"serving-guard-{platform}"]
        assert guard["uses"] == "hw-native-sys/pypto-serving/.github/workflows/ci.yml@main"
        assert guard["with"] == {
            "platform": platform,
            "pypto-repository": "${{ github.repository }}",
            "pypto-revision": "${{ needs.changes.outputs.checkout_sha }}",
        }
        assert guard["permissions"] == {"contents": "read"}
        assert "needs.changes.outputs.ptoas_version_changed == 'true'" in guard["if"]
    # The full aggregate itself is not the required context: it is attached to
    # main in dispatch runs. Only trusted control jobs write the PR status.
    assert workflow["permissions"] == {"contents": "read"}
    publisher = yaml.safe_load((ROOT / ".github/workflows/ci-full-status.yml").read_text())
    assert isinstance(publisher, dict)
    assert publisher["jobs"]["publish"]["permissions"]["statuses"] == "write"


def test_composite_checkout_retries_keep_the_pinned_revision():
    action = yaml.safe_load((ROOT / ".github/actions/setup-ci-job/action.yml").read_text())
    assert isinstance(action, dict)
    checkouts = [
        step for step in action["runs"]["steps"] if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkouts) == 2
    assert all(step["with"]["ref"] == "${{ inputs.checkout-ref }}" for step in checkouts)


@pytest.mark.parametrize("conflicting", [False, None])
def test_unavailable_merge_never_dispatches(control, monkeypatch, event, pr, merge_clock, conflicting):
    pr["mergeable"] = conflicting

    def request(path):
        assert path.endswith("/pulls/123")
        return pr

    monkeypatch.setattr(control, "api", request)
    with pytest.raises(ValueError, match="merge conflicts" if conflicting is False else "60 seconds"):
        control.dispatch(REPO, event)
    assert merge_clock == ([] if conflicting is False else [5] * 12)


@pytest.mark.parametrize("pending", ["unknown", "missing-merge", "stale-head"])
def test_command_waits_for_consistent_merge_without_authorizing_another_revision(
    control, monkeypatch, event, pr, merge_clock, pending
):
    current = copy.deepcopy(pr)
    reads = []
    new_merge = "d" * 40

    def request(path):
        reads.append(path)
        if path.endswith("/pulls/123"):
            if not merge_clock:
                if pending == "unknown":
                    current["mergeable"] = None
                elif pending == "missing-merge":
                    current["merge_commit_sha"] = None
            else:
                current.update(mergeable=True, merge_commit_sha=new_merge)
            return copy.deepcopy(current)
        if path.endswith(f"/git/commits/{MERGE}"):
            # A previous PR head must still wait for the current test merge.
            assert pending == "stale-head"
            return {"parents": [{"sha": BASE}, {"sha": "e" * 40}]}
        assert path.endswith(f"/git/commits/{new_merge}")
        return {"parents": [{"sha": BASE}, {"sha": HEAD}]}

    monkeypatch.setattr(control, "api", request)
    result = control.validated_request(REPO, event)
    assert result["head"]["sha"] == HEAD and result["base"]["sha"] == BASE
    assert result["merge_commit_sha"] == new_merge
    assert merge_clock == [5]
    assert reads.count(f"repos/{REPO}/pulls/123") == 2


@pytest.mark.parametrize("change", ["head", "base", "closed", "conflict"])
def test_merge_wait_stops_when_request_is_no_longer_valid(
    control, monkeypatch, event, pr, merge_clock, change
):
    if change == "base":
        monkeypatch.setattr(control, "current_base", lambda repo: "d" * 40 if merge_clock else BASE)

    def request(path):
        assert path.endswith("/pulls/123"), "Rejected revisions must not dispatch or write status"
        current = copy.deepcopy(pr)
        current["mergeable"] = None
        if merge_clock:
            if change in {"head", "base"}:
                current[change]["sha"] = "d" * 40
            elif change == "closed":
                current["state"] = "closed"
            else:
                current["mergeable"] = False
        return current

    monkeypatch.setattr(control, "api", request)
    with pytest.raises(ValueError, match="changed while waiting|open PR|merge conflicts"):
        control.dispatch(REPO, event)
    assert merge_clock == [5]


def test_inconsistent_merge_parents_exhaust_wait_without_dispatch(
    control, monkeypatch, event, pr, merge_clock
):
    def request(path):
        if path.endswith("/pulls/123"):
            return pr
        assert path.endswith(f"/git/commits/{MERGE}")
        return {"parents": [{"sha": BASE}, {"sha": "d" * 40}]}

    monkeypatch.setattr(control, "api", request)
    with pytest.raises(ValueError, match="60 seconds"):
        control.dispatch(REPO, event)
    assert merge_clock == [5] * 12


def test_merge_wait_does_not_retry_api_access_failures(control, monkeypatch, event, merge_clock):
    def denied(path):
        raise subprocess.CalledProcessError(1, ["gh", "api"], stderr="HTTP 403")

    monkeypatch.setattr(control, "api", denied)
    with pytest.raises(subprocess.CalledProcessError):
        control.validated_request(REPO, event)
    assert not merge_clock


def test_prepare_full_pins_merge_not_main(control, monkeypatch, tmp_path, pr):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_SHA", "e" * 40)

    def request(path):
        if path.endswith("/pulls/123"):
            return pr
        assert path.endswith(f"/git/commits/{MERGE}")
        return {"parents": [{"sha": BASE}, {"sha": HEAD}]}

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(control, "matching_lint_run", lambda *args: {"id": 40})
    control.prepare(
        REPO, {"inputs": {"pr_number": "123", "head_sha": HEAD, "base_sha": BASE, "merge_sha": MERGE}}
    )
    assert target.read_text() == f"full=true\ncheckout_sha={MERGE}\nreuse_lint=true\n"


def test_manual_full_on_untrusted_branch_is_rejected(control, monkeypatch):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/feature")
    monkeypatch.setenv("GITHUB_SHA", MERGE)
    with pytest.raises(ValueError, match="workflow_dispatch on main"):
        control.prepare(REPO, {"inputs": {"pr_number": "123"}})


def test_cancelled_run_cannot_reuse_successful_finalizer(control, publication):
    publication[0]["conclusion"] = "cancelled"
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert publication[3][0][1]["state"] == "failure"


def test_pagination_reads_every_job(control, monkeypatch):
    calls = []

    def request(path):
        calls.append(path)
        return {
            "jobs": [{"name": "matrix leg"}] * 100
            if path.endswith("&page=1")
            else [{"name": "full-result", "conclusion": "failure"}]
        }

    monkeypatch.setattr(control, "api", request)
    result = control.pages("repos/owner/repo/actions/runs/50/jobs?filter=latest", "jobs")
    assert len(result) == 101
    assert result[-1]["name"] == "full-result"
    assert len(calls) == 2


def test_forged_merge_with_correct_parents_is_rejected(control, pr):
    with pytest.raises(ValueError, match="current test merge"):
        control.verify_merge(REPO, pr, HEAD, BASE, "d" * 40)


@pytest.mark.parametrize("behind", [0, 1, 4, 5, 9, 10, 11, 12])
@pytest.mark.parametrize("ancestor", [True, False])
def test_result_freshness_allows_bounded_main_advance(controller, monkeypatch, behind, ancestor):
    current = "d" * 40
    monkeypatch.setattr(controller, "current_base", lambda repo: current)

    def api(path):
        if path == f"repos/{REPO}/compare/{current}...{HEAD}?per_page=1":
            return {"behind_by": behind}
        assert path == f"repos/{REPO}/compare/{BASE}...{current}?per_page=1"
        return {"merge_base_commit": {"sha": BASE if ancestor else "e" * 40}, "ahead_by": behind}

    monkeypatch.setattr(controller, "api", api)
    if behind > 10 or not ancestor:
        with pytest.raises(ValueError, match="rebase"):
            controller.verify_result_freshness(REPO, HEAD, BASE)
    else:
        controller.verify_result_freshness(REPO, HEAD, BASE)


@pytest.mark.parametrize("behind", [1, 4, 5, 9, 10, 11])
def test_publication_checks_freshness_and_explains_rejection(control, monkeypatch, publication, behind):
    calls = []

    def freshness(repo, head, base):
        calls.append((repo, head, base))
        if behind > 10:
            raise ValueError(f"PR is {behind} commits behind main; rebase and post /ci full again")

    monkeypatch.setattr(control, "verify_result_freshness", freshness)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    status = publication[3][0][1]
    assert calls == [(REPO, HEAD, BASE)]
    assert status["state"] == ("success" if behind <= 10 else "failure")
    if behind > 10:
        assert "11 commits behind main; rebase" in status["description"]


def test_running_full_stays_pending_when_main_advances(control, monkeypatch, publication):
    monkeypatch.setattr(control, "current_base", lambda repo: "d" * 40)
    publication[0].update(status="in_progress", conclusion=None)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert publication[3][0][1]["state"] == "pending"


def test_unavailable_base_comparison_cannot_publish_success(control, monkeypatch, publication):
    def unavailable(*args):
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "verify_result_freshness", unavailable)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert publication[3][0][1]["state"] == "failure"


def test_new_request_on_another_base_supersedes_old_run(control, publication):
    run, _, candidates, writes = publication
    newer = copy.deepcopy(run)
    newer.update(id=51, run_number=9, display_title=f"CI full PR #123 @{HEAD} base:{'d' * 40}")
    candidates.append(newer)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not writes


@pytest.mark.parametrize("status", ["queued", "in_progress", "requested", "waiting", "pending", "completed"])
@pytest.mark.parametrize("branch", ["main", "feature", "release-tag"])
def test_only_main_unfinished_runs_block_duplicate_dispatch(control, monkeypatch, event, pr, status, branch):
    monkeypatch.setattr(control, "matching_lint_run", lambda *args: {"id": 40})
    monkeypatch.setattr(control, "already_running_or_passed", lambda *args: False)
    queries = []
    writes = []

    def request(path, payload=None):
        if payload is not None:
            writes.append(payload)
            return {"workflow_run_id": 51} if path.endswith("/dispatches") else None
        if path.endswith("/pulls/123"):
            return pr
        if path.endswith(f"/git/commits/{MERGE}"):
            return {"parents": [{"sha": BASE}, {"sha": HEAD}]}
        queries.append(path)
        assert "&branch=main&created=" in path
        assert "status=" not in path, "Status transitions must not change query membership"
        return {
            "workflow_runs": [
                {
                    "display_title": f"CI full PR #123 @{HEAD} base:{BASE}",
                    "head_branch": branch,
                    "path": control.WORKFLOW,
                    "event": "workflow_dispatch",
                    "status": status,
                }
            ]
        }

    monkeypatch.setattr(control, "api", request)
    control.dispatch(REPO, event)
    assert len(queries) == 1
    assert len(writes) == (0 if branch == "main" and status != "completed" else 4)
    if writes:
        # An accepted full request also starts its informational progress monitor.
        assert writes[-1] == {"ref": "main", "inputs": {"run_id": "51"}}
        assert writes[0]["state"] == "pending"
        assert writes[1]["ref"] == "main"


def test_publication_history_is_bounded_by_creation_time(control, monkeypatch):
    calls = []

    def request(path):
        calls.append(path)
        assert "&branch=main" in path
        assert "&created=>=2026-11-05T01:00:00Z&per_page=100" in path
        if path.endswith("page=1"):
            # Unrelated runs can fill pages. Do not mistake them for requests
            # for this PR, or assume a short history that fits in one page.
            return {"workflow_runs": [{"display_title": "Manual main", "run_number": 90}] * 100}
        assert path.endswith("page=2")
        return {
            "workflow_runs": [
                {
                    "display_title": f"CI full PR #123 @{HEAD} base:{BASE}",
                    "run_number": 10,
                    "workflow_id": 99,
                    "path": control.WORKFLOW,
                    "head_branch": "main",
                }
            ]
        }

    monkeypatch.setattr(control, "api", request)
    assert control.newer_request_exists(
        REPO,
        123,
        HEAD,
        {"created_at": "2026-11-05T01:00:00Z", "run_number": 8, "workflow_id": 99, "path": control.WORKFLOW},
    )
    assert len(calls) == 2


@pytest.mark.parametrize("publication_scan", [False, True])
def test_filtered_run_search_cap_cannot_silently_truncate(control, monkeypatch, publication_scan):
    calls = []

    def request(path):
        calls.append(path)
        assert len(calls) <= 10, "The API cannot return more than 1,000 filtered runs"
        return {"workflow_runs": [{"display_title": "Manual main", "run_number": 90}] * 100}

    monkeypatch.setattr(control, "api", request)
    with pytest.raises(ValueError, match="pagination limit"):
        if publication_scan:
            control.newer_request_exists(
                REPO,
                123,
                HEAD,
                {
                    "created_at": "2026-11-05T01:00:00Z",
                    "run_number": 8,
                    "workflow_id": 99,
                    "path": control.WORKFLOW,
                },
            )
        else:
            control.pages(
                "repos/owner/repo/actions/workflows/ci-full.yml/runs?status=queued", "workflow_runs"
            )
    assert len(calls) == 10


@pytest.mark.parametrize("run_status", ["in_progress", "completed"])
def test_capped_scan_invalidates_the_rerun_previous_success(control, monkeypatch, publication, run_status):
    run, _, _, writes = publication
    run.update(status=run_status, conclusion="failure")

    def capped(*args):
        raise ValueError("Recent CI requests exceeded the pagination limit")

    def latest_status(path, key):
        assert path == f"repos/{REPO}/commits/{HEAD}/status"
        assert key == "statuses"
        return [
            {
                "context": "ci/full",
                "state": "success",
                "target_url": f"https://github.com/{REPO}/actions/runs/50/attempts/1",
            }
        ]

    monkeypatch.setattr(control, "newer_request_exists", capped)
    monkeypatch.setattr(control, "pages", latest_status)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert writes[0][1]["state"] == "failure"


@pytest.mark.parametrize("candidate_number", [7, 9])
def test_capped_scan_preserves_only_a_verified_newer_status(
    control, monkeypatch, publication, candidate_number
):
    run, _, _, writes = publication
    candidate = copy.deepcopy(run)
    candidate.update(id=51, run_number=candidate_number)
    original_api = control.api

    def capped(*args):
        raise ValueError("Recent CI requests exceeded the pagination limit")

    def request(path, payload=None):
        if path.endswith("/actions/runs/51"):
            return candidate
        return original_api(path, payload)

    monkeypatch.setattr(control, "newer_request_exists", capped)
    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {"context": "ci/full", "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1"}
        ],
    )
    control.publish(REPO, {"workflow_run": {"id": 50}})
    if candidate_number > run["run_number"]:
        assert not writes
    else:
        assert writes[0][1]["state"] == "failure"


@pytest.mark.parametrize("status_lookup_fails", [False, True])
def test_metadata_failure_cannot_preserve_previous_green(
    control, monkeypatch, publication, status_lookup_fails
):
    def request(path, key):
        if key == "statuses" and not status_lookup_fails:
            return []
        raise subprocess.CalledProcessError(1, ["gh", "api"], stderr="API unavailable")

    monkeypatch.setattr(control, "pages", request)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert publication[3][0][1]["state"] == "failure"


@pytest.mark.parametrize("lookup_path", ["/pulls/123", "/workflows/ci-full.yml"])
@pytest.mark.parametrize("run_status", ["in_progress", "completed"])
@pytest.mark.parametrize("lookup_error", ["http", "json"])
def test_gate_lookup_failure_invalidates_previous_success(
    control, monkeypatch, publication, run_status, lookup_error, lookup_path
):
    run, _, _, writes = publication
    run.update(status=run_status, conclusion="failure")
    original_api = control.api

    def request(path, payload=None):
        if path.endswith(lookup_path):
            if lookup_error == "http":
                raise subprocess.CalledProcessError(1, ["gh", "api"], stderr="Metadata lookup unavailable")
            raise ValueError("Invalid metadata JSON")
        return original_api(path, payload)

    def latest_status(path, key):
        assert path == f"repos/{REPO}/commits/{HEAD}/status"
        assert key == "statuses"
        return [
            {
                "context": "ci/full",
                "state": "success",
                "target_url": f"https://github.com/{REPO}/actions/runs/50/attempts/1",
            }
        ]

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(control, "pages", latest_status)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert len(writes) == 1
    assert writes[0][0] == f"repos/{REPO}/statuses/{HEAD}"
    assert writes[0][1]["state"] == "failure"


@pytest.mark.parametrize("lookup_path", ["/pulls/123", "/workflows/ci-full.yml"])
def test_gate_lookup_failure_preserves_verified_newer_status(control, monkeypatch, publication, lookup_path):
    run, _, _, writes = publication
    candidate = copy.deepcopy(run)
    candidate.update(id=51, run_number=9)
    original_api = control.api

    def request(path, payload=None):
        if path.endswith(lookup_path):
            raise subprocess.CalledProcessError(1, ["gh", "api"], stderr="Metadata lookup unavailable")
        if path.endswith("/actions/runs/51"):
            return candidate
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {"context": "ci/full", "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1"}
        ],
    )
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not writes


@pytest.mark.parametrize("cli_error", [False, True])
def test_expected_errors_reach_summary_and_exit_nonzero(control, monkeypatch, tmp_path, cli_error):
    event_file, summary = tmp_path / "event.json", tmp_path / "summary"
    event_file.write_text(json.dumps({}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setenv("GH_TOKEN", "secret-token")
    output_file = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setattr(control.sys, "argv", ["ci_control.py", "dispatch"])

    def fail(*args):
        if cli_error:
            raise subprocess.CalledProcessError(1, ["gh", "api"], stderr="Forbidden <detail> secret-token")
        raise ValueError("PR changed; post /ci full again for the new revision")

    monkeypatch.setattr(control, "dispatch", fail)
    with pytest.raises(SystemExit) as error:
        control.main()
    assert error.value.code == 1
    text = summary.read_text()
    public_error = json.loads(output_file.read_text().strip().split("=", 1)[1])
    assert "secret-token" not in public_error
    assert "dispatch failed:" in text
    if cli_error:
        assert "Forbidden &lt;detail&gt; [REDACTED]" in text
        assert "secret-token" not in text
        assert "GitHub API validation or dispatch failed" in public_error
    else:
        assert "PR changed" in text
        assert "PR changed" in public_error


@pytest.mark.parametrize(
    "reason", [None, "PR has merge conflicts; update the branch", "<unsafe>\nsecond line"]
)
@pytest.mark.parametrize("previous_author", [None, "github-actions[bot]", "external-user"])
def test_failed_command_feedback_links_logs_and_deduplicates_its_own_attempt(
    control, monkeypatch, event, reason, previous_author
):
    monkeypatch.setenv("GITHUB_RUN_ID", "50")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("CI_COMMAND_ERROR", json.dumps(reason) if reason is not None else "")
    marker = "<!-- ci-command-feedback:50:2 -->"
    existing = [] if previous_author is None else [{"user": {"login": previous_author}, "body": marker}]
    monkeypatch.setattr(control, "pages", lambda path: existing)
    writes = []
    monkeypatch.setattr(control, "api", lambda path, payload: writes.append((path, payload)))
    control.command_feedback(REPO, event)
    if previous_author == "github-actions[bot]":
        assert not writes
    else:
        assert len(writes) == 1 and writes[0][0] == f"repos/{REPO}/issues/123/comments"
        body = writes[0][1]["body"]
        assert marker in body and f"https://github.com/{REPO}/actions/runs/50/attempts/2" in body
        assert "could not be completed" in body
        assert "<unsafe>" not in body
        if reason == "<unsafe>\nsecond line":
            assert "&lt;unsafe&gt;\nsecond line" in body


@pytest.mark.parametrize("invalid", ["edited", "issue", "other-command"])
def test_feedback_ignores_non_commands(control, monkeypatch, event, invalid):
    if invalid == "edited":
        event["action"] = "edited"
    elif invalid == "issue":
        del event["issue"]["pull_request"]
    else:
        event["comment"]["body"] = "/ci full extra"
    monkeypatch.setattr(control, "pages", lambda *args: pytest.fail("No comment reads"))
    control.command_feedback(REPO, event)


def test_command_feedback_permission_is_separate_from_routing_and_dispatch():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci-command.yml").read_text())
    jobs = workflow["jobs"]
    feedback = jobs["feedback"]
    assert feedback["needs"] == ["route", "dispatch"]
    assert "always()" in feedback["if"]
    assert "needs.route.result == 'failure'" in feedback["if"]
    assert "needs.dispatch.result == 'failure'" in feedback["if"]
    assert feedback["permissions"] == {"contents": "read", "pull-requests": "write"}
    assert feedback["runs-on"] == "ubuntu-latest"
    assert feedback["steps"][0]["with"]["ref"] == "${{ github.workflow_sha }}"
    assert feedback["steps"][-1]["run"].endswith("ci_control.py command-feedback")
    assert "CI_COMMAND_ERROR" in feedback["steps"][-1]["env"]
    for name in ("route", "dispatch"):
        assert "issues" not in jobs[name]["permissions"]
        assert jobs[name]["permissions"]["pull-requests"] == "read"
        assert jobs[name]["outputs"]["error"] == f"${{{{ steps.{name}.outputs.command_error }}}}"


def test_pr_jobs_cannot_write_github_caches():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci-full.yml").read_text())
    assert workflow["cache-mode"] == "read"
    for job in workflow["jobs"].values():
        assert job.get("cache-mode", "read") in {"read", "none"}
    producer = workflow["jobs"]["prebuild-wheels-github"]
    upload = next(s for s in producer["steps"] if s.get("uses", "").startswith("actions/upload-artifact@"))
    assert "github.run_attempt" in upload["with"]["name"]
    for name in ("unit-tests", "system-tests-a5sim"):
        downloads = [
            s
            for s in workflow["jobs"][name]["steps"]
            if s.get("uses", "").startswith("actions/download-artifact@")
        ]
        assert len(downloads) == 1
        assert downloads[0]["with"]["name"] == "${{ needs.prebuild-wheels-github.outputs.artifact_name }}"
        assert "run-id" not in downloads[0]["with"]


def test_full_rerun_concurrency_is_scoped_to_original_request():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci-full.yml").read_text())
    assert "format('full-run-{0}', github.run_id)" in workflow["concurrency"]["group"]
    publisher = yaml.safe_load((ROOT / ".github/workflows/ci-full-status.yml").read_text())
    assert (
        publisher["jobs"]["publish"]["concurrency"]["group"]
        == "ci-full-status-${{ needs.route.outputs.key }}"
    )
    route_job = publisher["jobs"]["route"]
    assert "concurrency" not in route_job
    assert "github.event.workflow_run.head_branch == 'main'" in route_job["if"]
    assert "display_title" not in route_job["if"]
    assert route_job["permissions"]["actions"] == "read"
    assert publisher[True]["workflow_dispatch"]["inputs"]["run_id"]["required"] is True
    assert publisher["jobs"]["publish"]["steps"][-1]["env"]["CI_ROUTED_EVENT"] == (
        "${{ needs.route.outputs.event }}"
    )
    assert route_job["steps"][0]["with"]["ref"] == "${{ github.workflow_sha }}"
    assert route_job["steps"][1]["run"] == "python .github/scripts/ci_control.py route"
    assert publisher["jobs"]["publish"]["needs"] == "route"
    assert publisher["jobs"]["publish"]["if"] == "needs.route.outputs.key != ''"
    command = yaml.safe_load((ROOT / ".github/workflows/ci-command.yml").read_text())
    dispatcher = command["jobs"]["dispatch"]
    assert dispatcher["permissions"]["statuses"] == "write"
    assert dispatcher["concurrency"] == publisher["jobs"]["publish"]["concurrency"]
    assert dispatcher["concurrency"]["queue"] == "max"
    assert dispatcher["concurrency"]["cancel-in-progress"] is False
    assert dispatcher["needs"] == "route"
    assert dispatcher["if"] == "needs.route.outputs.key != ''"
    assert dispatcher["steps"][-1]["env"]["CI_ROUTED_HEAD"] == "${{ needs.route.outputs.head }}"
    assert "concurrency" not in command["jobs"]["route"]
    assert command["jobs"]["route"]["permissions"] == {"contents": "read", "pull-requests": "read"}


def test_tiered_pr_controller_uses_trusted_base_without_changing_trigger():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    # PyYAML's YAML 1.1 loader parses the unquoted Actions key "on" as True.
    assert "pull_request" in workflow[True]
    controller = workflow["jobs"]["changes"]["steps"][0]["with"]
    assert controller["ref"] == (
        "${{ github.event_name == 'pull_request' && "
        "github.event.pull_request.base.sha || github.workflow_sha }}"
    )
    assert controller["sparse-checkout"] == ".github/scripts"
    assert controller["persist-credentials"] is False
    # The trusted base may still contain the old flag-based prepare helper.
    # It must also select lint before this PR merges, without a vars lookup.
    prepare = workflow["jobs"]["changes"]["steps"][1]
    assert prepare["env"]["CI_COMMENT_TIERS_ENABLED"] == "true"


@pytest.mark.parametrize(
    "case,event_name,expected",
    [
        ("docs", "pull_request", ("true", "true", "false")),
        ("docs", "workflow_dispatch", ("false", "true", "false")),
        ("mixed", "workflow_dispatch", ("false", "true", "false")),
        ("rename", "pull_request", ("false", "true", "false")),
        ("version", "workflow_dispatch", ("false", "false", "true")),
        ("same-version", "workflow_dispatch", ("false", "false", "false")),
        ("299-files", "pull_request", ("true", "true", "false")),
        ("300-files", "pull_request", ("false", "true", "true")),
        ("missing-files", "workflow_dispatch", ("false", "true", "true")),
        ("empty", "workflow_dispatch", ("false", "true", "true")),
        ("unavailable", "workflow_dispatch", ("false", "true", "true")),
        ("missing-version", "workflow_dispatch", ("false", "true", "true")),
    ],
)
def test_classification_uses_pinned_comparison(tmp_path, case, event_name, expected):
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci-full.yml").read_text())
    classify = next(s for s in workflow["jobs"]["changes"]["steps"] if s.get("id") == "classify")
    assert "inputs.base_sha" in classify["env"]["PR_BASE_SHA"]
    assert "inputs.head_sha" in classify["env"]["PR_HEAD_SHA"]
    files = [{"filename": "docs/en/example.md"}]
    if case == "mixed":
        files.insert(0, {"filename": "src/example.cpp"})
    elif case == "rename":
        files[0]["previous_filename"] = "src/example.cpp"
    elif case in {"version", "same-version", "missing-version"}:
        files = [{"filename": "toolchain/versions.env"}]
    elif case in {"299-files", "300-files"}:
        files *= int(case.split("-")[0])
    elif case == "empty":
        files = []
    comparison = {} if case == "missing-files" else {"files": files}
    output_file, calls_file = tmp_path / "outputs", tmp_path / "calls"
    env = dict(
        os.environ,
        GITHUB_OUTPUT=str(output_file),
        EVENT_NAME=event_name,
        PR_NUMBER="123",
        PR_BASE_SHA=BASE,
        PR_HEAD_SHA=HEAD,
        REPO=REPO,
        CI_TEST_CALLS=str(calls_file),
        CI_TEST_COMPARISON=json.dumps(comparison),
        CI_TEST_CASE=case,
    )
    # The stub rejects all mutable PR reads. Only the authorized immutable
    # comparison and the two pinned version-file requests can succeed.
    stub = r"""
    gh() {
      local endpoint="${!#}"
      printf '%s\n' "$endpoint" >> "$CI_TEST_CALLS"
      case "$endpoint" in
        "repos/$REPO/compare/$PR_BASE_SHA...$PR_HEAD_SHA?per_page=1")
          [ "$CI_TEST_CASE" != unavailable ] || return 1
          printf '%s\n' "$CI_TEST_COMPARISON" ;;
        "repos/$REPO/contents/toolchain/versions.env?ref=$PR_BASE_SHA")
          printf 'PTOAS_VERSION=1\n' ;;
        "repos/$REPO/contents/toolchain/versions.env?ref=$PR_HEAD_SHA")
          [ "$CI_TEST_CASE" != missing-version ] || return 1
          if [ "$CI_TEST_CASE" = same-version ]; then
            printf '# changed comment\nPTOAS_VERSION=1\n'
          else
            printf 'PTOAS_VERSION=2\n'
          fi ;;
        *) return 99 ;;
      esac
    }
    """
    result = subprocess.run(
        ["bash", "-c", stub + classify["run"]], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in output_file.read_text().splitlines())
    assert tuple(outputs[k] for k in ("docs_only", "docs_changed", "ptoas_version_changed")) == expected
    calls = calls_file.read_text().splitlines()
    assert calls[0] == f"repos/{REPO}/compare/{BASE}...{HEAD}?per_page=1"
    assert all("/pulls/" not in call for call in calls)


@pytest.fixture
def publication_event(publication):
    run = copy.deepcopy(publication[0])
    # Even a delayed successful completion must only invalidate on lookup failure.
    run.update(run_attempt=1, status="completed", conclusion="success")
    return {
        "action": "completed",
        "repository": copy.deepcopy(run["repository"]),
        "workflow": {"id": run["workflow_id"], "path": run["path"]},
        "workflow_run": run,
    }


@pytest.mark.parametrize("marker", ["", " (request 900-1)", " (request 900-2)"])
@pytest.mark.parametrize("lookup_fails", [False, True])
def test_unbound_request_marker_survives_old_publisher(
    control, monkeypatch, publication, publication_event, lookup_fails, marker
):
    original_api = control.api

    def request(path, payload=None):
        if lookup_fails and path.endswith("/actions/runs/50"):
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {
                "context": "ci/full",
                "state": "pending",
                "description": f"Full CI requested; waiting for dispatch{marker}",
                "target_url": f"https://github.com/{REPO}/pull/123",
            }
        ],
    )
    # No replacement run exists yet (including a failed/lost dispatch).
    control.publish(REPO, publication_event)
    assert not publication[3]


@pytest.mark.parametrize("failure", ["bind", "ack", "lost_response"])
@pytest.mark.parametrize("result", ["success", "failure"])
def test_accepted_full_recovers_unbound_request(
    control, monkeypatch, pr, event, publication, failure, result
):
    run, jobs, _, writes = publication
    run["conclusion"] = jobs[0]["conclusion"] = result
    gate = {}
    original_api, original_pages = control.api, control.pages
    monkeypatch.setattr(control, "validated_request", lambda *args: pr)
    monkeypatch.setattr(control, "matching_lint_run", lambda *args: {"id": 40})
    monkeypatch.setattr(control, "already_running_or_passed", lambda *args: False)
    monkeypatch.setattr(control, "unfinished_request_exists", lambda *args: False)
    monkeypatch.setattr(control, "verify_merge", lambda *args: BASE)

    def api(path, payload=None):
        if payload is not None and path.endswith(f"/statuses/{HEAD}"):
            if payload["description"] == "Full CI dispatched; waiting for results":
                assert failure == "bind"
                raise subprocess.CalledProcessError(1, ["gh", "api"])
            gate.update(payload)
        if payload is not None and path.endswith("ci-full.yml/dispatches"):
            run["display_title"] += f" request:{payload['inputs']['request_id']}"
            if failure == "lost_response":
                raise subprocess.CalledProcessError(1, ["gh", "api"])
            return None if failure == "ack" else {"workflow_run_id": 50}
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setattr(
        control, "pages", lambda path, key: [gate] if key == "statuses" else original_pages(path, key)
    )
    with pytest.raises(ValueError if failure == "ack" else subprocess.CalledProcessError):
        control.dispatch(REPO, event)
    assert gate["state"] == "pending" and "request 900-1" in gate["description"]
    # Completion and explicit reconciliation use the same publication path.
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert gate["state"] == result
    assert gate["target_url"].endswith("/runs/50/attempts/2")
    # A new command attempt must still exclude this now-completed source run.
    gate.update(
        state="pending",
        description="Full CI requested; waiting for dispatch (request 900-2)",
        target_url=f"https://github.com/{REPO}/pull/123",
    )
    before = len(writes)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert len(writes) == before and gate["state"] == "pending"


def test_bound_replacement_survives_stale_run_history(control, monkeypatch, publication):
    run, _, _, writes = publication
    newer = copy.deepcopy(run)
    newer.update(id=51, run_number=9, run_attempt=1)
    original_api = control.api

    def request(path, payload=None):
        if path.endswith("/actions/runs/51"):
            return newer
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {
                "context": "ci/full",
                "state": "pending",
                "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1",
            }
        ],
    )
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not writes


@pytest.mark.parametrize("action", ["requested", "in_progress", "completed"])
def test_verified_event_routes_all_attempts_and_bases_to_same_head(
    control, monkeypatch, publication_event, tmp_path, action
):
    output_file = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    def lookup(path):
        assert path == (
            f"repos/{REPO}/actions/runs/50/attempts/{publication_event['workflow_run']['run_attempt']}"
        )
        return copy.deepcopy(publication_event["workflow_run"])

    monkeypatch.setattr(control, "api", lookup)
    publication_event["action"] = action
    control.route(REPO, publication_event)
    publication_event["workflow_run"].update(
        run_attempt=2, display_title=f"CI full PR #123 @{HEAD} base:{'d' * 40}"
    )
    control.route(REPO, publication_event)
    lines = output_file.read_text().splitlines()
    assert lines[::2] == [f"key=123-{HEAD}"] * 2
    assert all(json.loads(line.removeprefix("event="))["workflow_run"]["id"] == 50 for line in lines[1::2])


@pytest.mark.parametrize("action", ["requested", "in_progress", "completed"])
@pytest.mark.parametrize("lookup_error", ["http", "json"])
def test_live_lookup_failure_invalidates_event_success(
    control, monkeypatch, publication, publication_event, action, lookup_error
):
    publication_event["action"] = action
    original_api = control.api

    def request(path, payload=None):
        if path.endswith("/actions/runs/50"):
            if lookup_error == "http":
                raise subprocess.CalledProcessError(1, ["gh", "api"], stderr="Live run unavailable")
            raise ValueError("Invalid run JSON")
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {"context": "ci/full", "target_url": f"https://github.com/{REPO}/actions/runs/50/attempts/1"}
        ],
    )
    control.publish(REPO, publication_event)
    writes = publication[3]
    assert len(writes) == 1
    assert writes[0][0] == f"repos/{REPO}/statuses/{HEAD}"
    assert writes[0][1]["state"] == "failure"


@pytest.mark.parametrize(
    "path,value",
    [
        (("action",), "unrelated"),
        (("repository", "full_name"), "unrelated/repo"),
        (("repository", "id"), 3),
        (("workflow", "path"), ".github/workflows/fake.yml"),
        (("workflow", "id"), 100),
        (("workflow_run", "path"), ".github/workflows/fake.yml"),
        (("workflow_run", "head_branch"), "feature"),
        (("workflow_run", "event"), "pull_request"),
        (("workflow_run", "repository", "full_name"), "unrelated/repo"),
        (("workflow_run", "head_repository", "id"), 2),
        (("workflow_run", "run_attempt"), 0),
        (("workflow_run", "run_number"), "8"),
        (("workflow_run", "workflow_id"), None),
        (("workflow",), None),
    ],
)
@pytest.mark.parametrize("operation", ["publish", "route"])
def test_untrusted_event_cannot_publish_or_join_status_queue(
    control, monkeypatch, publication, publication_event, path, value, operation, tmp_path
):
    output_file = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    target = publication_event
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    def unavailable(path, payload=None):
        assert payload is None
        assert path.endswith("/actions/runs/50")
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "api", unavailable)
    with pytest.raises(control.UntrustedRunError, match="event provenance"):
        getattr(control, operation)(REPO, publication_event)
    assert not publication[3]
    assert not output_file.exists()


@pytest.mark.parametrize("newer_lookup_fails", [False, True])
def test_live_lookup_failure_preserves_only_verified_newer_writer(
    control, monkeypatch, publication, publication_event, newer_lookup_fails
):
    candidate = copy.deepcopy(publication[0])
    candidate.update(id=51, run_number=9)
    original_api = control.api

    def request(path, payload=None):
        if path.endswith("/actions/runs/50") or (newer_lookup_fails and path.endswith("/actions/runs/51")):
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        if path.endswith("/actions/runs/51"):
            return candidate
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {"context": "ci/full", "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1"}
        ],
    )
    control.publish(REPO, publication_event)
    if newer_lookup_fails:
        assert publication[3][0][1]["state"] == "failure"
    else:
        assert not publication[3]


@pytest.mark.parametrize("action", ["requested", "in_progress", "completed"])
@pytest.mark.parametrize("state", ["pending", "success", "failure"])
def test_delayed_event_preserves_verified_later_attempt(
    control, monkeypatch, publication, publication_event, action, state
):
    publication_event["action"] = action
    candidate = copy.deepcopy(publication[0])
    candidate.update(status="in_progress" if state == "pending" else "completed", conclusion=state)
    calls = []
    original_api = control.api

    def request(path, payload=None):
        calls.append(path)
        if path.endswith("/actions/runs/50"):
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        if path.endswith("/actions/runs/50/attempts/2"):
            return candidate
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {
                "context": "ci/full",
                "state": state,
                "target_url": f"https://github.com/{REPO}/actions/runs/50/attempts/2",
            }
        ],
    )
    control.publish(REPO, publication_event)
    assert f"repos/{REPO}/actions/runs/50/attempts/2" in calls
    assert not publication[3]


@pytest.mark.parametrize(
    "case",
    [
        "http",
        "json",
        "id",
        "run_attempt",
        "run_number",
        "workflow_id",
        "head_branch",
        "event",
        "display_title",
        "path",
        "repository",
        "head_repository",
    ],
)
def test_unverified_later_attempt_cannot_preserve_success(
    control, monkeypatch, publication, publication_event, case
):
    candidate = copy.deepcopy(publication[0])
    if case in {"id", "run_attempt", "run_number", "workflow_id"}:
        candidate[case] += 1
    elif case in {"repository", "head_repository"}:
        candidate[case]["id"] = 3
    elif case not in {"http", "json"}:
        candidate[case] = "unrelated"
    original_api = control.api

    def request(path, payload=None):
        if path.endswith("/actions/runs/50"):
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        if path.endswith("/actions/runs/50/attempts/2"):
            if case == "http":
                raise subprocess.CalledProcessError(1, ["gh", "api"])
            if case == "json":
                raise ValueError("Invalid attempt JSON")
            return candidate
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {
                "context": "ci/full",
                "state": "success",
                "target_url": f"https://github.com/{REPO}/actions/runs/50/attempts/2",
            }
        ],
    )
    control.publish(REPO, publication_event)
    assert len(publication[3]) == 1
    assert publication[3][0][1]["state"] == "failure"


@pytest.mark.parametrize("suffix", ["", "/attempts/1", "/attempts/2", "/attempts/invalid"])
def test_same_or_older_attempt_url_cannot_preserve_success(control, monkeypatch, publication, suffix):
    run = publication[0]  # The live attempt is 2.
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {"context": "ci/full", "target_url": f"https://github.com/{REPO}/actions/runs/50{suffix}"}
        ],
    )
    monkeypatch.setattr(control, "api", lambda *args: pytest.fail("No newer attempt to verify"))
    assert not control.newer_status_exists(REPO, 123, HEAD, run)


@pytest.mark.parametrize("http_status", [404, 403, 503])
def test_a5_missing_owner_requires_persisted_pr_identity(a5_control, a5_request, monkeypatch, http_status):
    _, event, writes, outputs = a5_request

    def check(check_id, key, status="in_progress", slug="github-actions"):
        return {"id": check_id, "external_id": key, "status": status, "app": {"slug": slug}}

    checks = [
        check(81, "a5-pr-123:44"),
        check(82, "a5-pr-123:44:2:81"),
        check(83, "44:3:81"),
        check(84, "a5-pr-124:44"),
        check(85, "44"),
        check(86, "a5-pr-123:44:4:81", "completed"),
        check(87, "a5-pr-123:44", slug="other-app"),
        check(88, "a5-pr-123:44:bad"),
    ]
    api = a5_control.api

    def missing(path, *args, **kwargs):
        if path.endswith("/actions/runs/44"):
            raise subprocess.CalledProcessError(1, "gh", stderr=f"gh: error (HTTP {http_status})")
        return api(path, *args, **kwargs)

    monkeypatch.setattr(a5_control, "api", missing)
    monkeypatch.setattr(
        a5_control, "head_checks", lambda repo, head: checks if repo == REPO and head == HEAD else []
    )
    if http_status != 404:
        with pytest.raises(subprocess.CalledProcessError):
            a5_control.prepare(REPO, event)
        assert not writes and not outputs
        return
    a5_control.prepare(REPO, event)
    assert [(path, payload["conclusion"], kwargs["method"]) for path, payload, kwargs in writes[:-1]] == [
        (f"repos/{REPO}/check-runs/{i}", "cancelled", "PATCH") for i in (81, 82)
    ]
    assert writes[-1][1]["external_id"] == "a5-pr-123:1234"
    assert outputs == {"checkout_sha": MERGE, "check_id": "987"}


@pytest.mark.parametrize("unfinished", [False, True])
def test_duplicate_scan_splits_large_history_without_losing_old_reruns(control, monkeypatch, unfinished):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(seconds=1100)
    title = f"CI full PR #123 @{HEAD} base:{BASE}"
    calls = []
    runs = [
        {
            "display_title": title,
            "head_branch": "main",
            "event": "workflow_dispatch",
            "path": control.WORKFLOW,
            "status": "requested" if unfinished and second == 0 else "completed",
            "created_at": f"{start + timedelta(seconds=second):%Y-%m-%dT%H:%M:%SZ}",
        }
        for second in range(1101)
    ]

    def request(path):
        query = parse_qs(urlsplit(path).query)
        assert "status" not in query
        lower, upper = (
            datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            for v in query["created"][0].split("..")
        )
        page = int(query["page"][0])
        assert 1 <= page <= 10
        calls.append((lower, upper, page))
        # A rerun created at the oldest boundary can change state between pages.
        # Its membership must not depend on which status was queried first.
        if unfinished and len(calls) > 1:
            runs[0]["status"] = "queued"
        matches = [
            run
            for run in reversed(runs)
            if f"{lower:%Y-%m-%dT%H:%M:%SZ}" <= run["created_at"] <= f"{upper:%Y-%m-%dT%H:%M:%SZ}"
        ]
        return {"workflow_runs": matches[(page - 1) * 100 : page * 100]}

    monkeypatch.setattr(control, "api", request)
    assert control.unfinished_request_exists(REPO, title, start, end) is unfinished
    windows = {(lower, upper) for lower, upper, _ in calls}
    middle = start + timedelta(seconds=550)
    assert windows == {(start, end), (start, middle), (middle + timedelta(seconds=1), end)}


def test_duplicate_scan_rejects_unsplittable_cap(control, monkeypatch):
    point = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(control, "api", lambda *args: {"workflow_runs": [{"display_title": "Other"}] * 100})
    with pytest.raises(ValueError, match="cannot verify duplicates"):
        control.unfinished_request_exists(REPO, "Full request", point, point)


def test_duplicate_scan_propagates_api_failure(control, monkeypatch):
    point = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def unavailable(*args):
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "api", unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        control.unfinished_request_exists(REPO, "Full request", point, point)


@pytest.fixture
def lint_run(control, monkeypatch, pr):
    run = {
        "id": 40,
        "run_number": 4,
        "run_attempt": 1,
        "status": "completed",
        "display_title": f"CI lint PR #123 @{HEAD} base:{BASE}",
        "event": "pull_request",
        "path": control.LINT_WORKFLOW,
        "workflow_id": 99,
        "head_sha": HEAD,
        "repository": {"full_name": REPO},
        "head_repository": {"id": 2},
    }
    jobs = [
        {"name": name, "status": "completed", "conclusion": "success"}
        for name in ("changes", "pre-commit", "clang-tidy")
    ]
    runs = [run]
    files = [{"filename": "docs/en/dev/index.md"}]

    def api(path, payload=None):
        assert payload is None, "Lint reuse cannot write to GitHub"
        if "/compare/" in path:
            assert path == f"repos/{REPO}/compare/{'d' * 40}...{BASE}?per_page=1"
            return {"merge_base_commit": {"sha": "d" * 40}}
        assert path == f"repos/{REPO}/actions/workflows/ci.yml"
        return {"id": 99}

    def pages(path, key=None):
        if path == f"repos/{REPO}/actions/workflows/ci.yml/runs?event=pull_request&head_sha={HEAD}":
            return runs
        if path == f"repos/{REPO}/actions/runs/40/attempts/1/jobs":
            return jobs
        assert path == f"repos/{REPO}/pulls/123/files"
        return files

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setattr(control, "pages", pages)
    return run, runs, jobs, files


def test_lint_reuse_accepts_matching_revision(control, pr, lint_run):
    assert control.matching_lint_run(REPO, pr)["id"] == 40


@pytest.mark.parametrize(
    "field,value",
    [
        ("display_title", f"CI lint PR #123 @{HEAD} base:invalid"),
        ("display_title", f"CI lint PR #124 @{HEAD} base:{BASE}"),
        ("event", "push"),
        ("path", ".github/workflows/other.yml"),
        ("workflow_id", 100),
        ("head_sha", "d" * 40),
        ("head_repository", {"id": 3}),
        ("repository", {"full_name": "other/repo"}),
    ],
)
def test_lint_reuse_rejects_unrelated_runs(control, pr, lint_run, field, value):
    lint_run[0][field] = value
    with pytest.raises(ValueError, match="No matching PR lint run"):
        control.matching_lint_run(REPO, pr)


@pytest.mark.parametrize("name", ["changes", "pre-commit", "clang-tidy"])
@pytest.mark.parametrize("result", ["failure", "cancelled", "timed_out", "in_progress", "missing"])
def test_lint_reuse_requires_complete_success(control, pr, lint_run, name, result):
    jobs = lint_run[2]
    job = next(job for job in jobs if job["name"] == name)
    if result == "missing":
        jobs.remove(job)
    elif result == "in_progress":
        job["status"] = result
    else:
        job["conclusion"] = result
    with pytest.raises(ValueError, match="PR lint job"):
        control.matching_lint_run(REPO, pr)


def test_lint_reuse_does_not_fall_back_to_old_green(control, pr, lint_run):
    lint_run[1].append({**lint_run[0], "run_number": 3, "id": 39})
    lint_run[2][1]["conclusion"] = "failure"
    with pytest.raises(ValueError, match="pre-commit did not pass"):
        control.matching_lint_run(REPO, pr)


@pytest.mark.parametrize("result", ["success", "failure", "cancelled"])
def test_latest_lint_wins_across_base_snapshots(control, pr, lint_run, result):
    run, runs, jobs, _ = lint_run
    # #3404: REST names the older cancelled run's base, while the newest
    # successful run records a different pull_request event snapshot.
    runs.append({**run, "id": 39, "run_number": 3})
    run["display_title"] = f"CI lint PR #123 @{HEAD} base:{'d' * 40}"
    jobs[1]["conclusion"] = result
    if result == "success":
        assert control.matching_lint_run(REPO, pr)["id"] == 40
    else:
        with pytest.raises(ValueError, match="pre-commit did not pass"):
            control.matching_lint_run(REPO, pr)


def test_bound_pending_survives_main_advance(control, monkeypatch, pr, bound_pending):
    monkeypatch.setattr(control, "current_base", lambda repo: "d" * 40)
    assert control.already_running_or_passed(REPO, pr)


@pytest.mark.parametrize("state", ["failure", "error", "pending", "success"])
@pytest.mark.parametrize("status", ["queued", "in_progress", "completed"])
def test_comment_during_native_rerun_after_main_advance(
    control, monkeypatch, pr, event, bound_pending, state, status
):
    run, gate = bound_pending
    run.update(run_attempt=3, status=status, conclusion="failure" if status == "completed" else None)
    gate["state"] = state  # The native rerun start event may not have updated it.
    monkeypatch.setattr(control, "current_base", lambda repo: "d" * 40)
    if status == "completed":
        assert not control.already_running_or_passed(REPO, pr)
    else:
        monkeypatch.setattr(control, "validated_request", lambda *args: pr)
        monkeypatch.setattr(
            control, "matching_lint_run", lambda *args: pytest.fail("A live rerun must stop dispatch")
        )
        control.dispatch(REPO, event)


@pytest.mark.parametrize("comparison", [{"merge_base_commit": {"sha": "e" * 40}}, {}, "api_failure"])
def test_lint_rejects_rewritten_or_unverifiable_base(control, monkeypatch, pr, lint_run, comparison):
    lint_run[0]["display_title"] = f"CI lint PR #123 @{HEAD} base:{'d' * 40}"
    original_api = control.api

    def api(path, payload=None):
        if "/compare/" in path:
            if comparison == "api_failure":
                raise subprocess.CalledProcessError(1, ["gh", "api"])
            return comparison
        return original_api(path, payload)

    monkeypatch.setattr(control, "api", api)
    with pytest.raises(subprocess.CalledProcessError if comparison == "api_failure" else ValueError):
        control.matching_lint_run(REPO, pr)


def test_lint_ignores_orphaned_history_but_keeps_latest_failure(control, monkeypatch, pr, lint_run):
    run, runs, jobs, _ = lint_run
    runs.append(
        {**run, "id": 41, "run_number": 5, "display_title": f"CI lint PR #123 @{HEAD} base:{'d' * 40}"}
    )
    runs.append({**run, "id": 39, "run_number": 3})
    original_api = control.api
    monkeypatch.setattr(
        control,
        "api",
        lambda path, payload=None: {"merge_base_commit": {"sha": "e" * 40}}
        if "/compare/" in path
        else original_api(path, payload),
    )
    assert control.matching_lint_run(REPO, pr)["id"] == 40
    jobs[1]["conclusion"] = "failure"
    with pytest.raises(ValueError, match="pre-commit did not pass"):
        control.matching_lint_run(REPO, pr)


def test_green_result_survives_allowed_main_advance(control, monkeypatch, pr, publication):
    api = control.api
    monkeypatch.setattr(control, "current_base", lambda repo: "d" * 40)
    monkeypatch.setattr(
        control,
        "api",
        lambda path, payload=None: {"merge_base_commit": {"sha": BASE}, "ahead_by": 1}
        if "/compare/" in path
        else api(path, payload),
    )
    run, _, _, writes = publication
    # Use the current bound attempt for successful-result deduplication.
    run["run_attempt"] = 1
    pages = control.pages
    monkeypatch.setattr(
        control, "pages", lambda path, key: publication[1] if key == "jobs" else pages(path, key)
    )
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert writes[0][1]["state"] == "success"
    assert control.already_running_or_passed(REPO, pr)


@pytest.mark.parametrize(
    "files,accepted",
    [
        ([{"filename": "README.md"}, {"filename": "mkdocs.yml"}], True),
        ([{"filename": "docs/new.md", "previous_filename": "src/deleted.cpp"}], False),
        ([{"filename": "docs/new.md", "previous_filename": "docs/old.md"}], True),
        ([{"filename": "python/changed.py"}], False),
        ([], False),
        ([{"filename": "docs/page.md"}] * 3000, False),
    ],
)
def test_lint_reuse_skipped_clang_requires_docs_only(control, pr, lint_run, files, accepted):
    lint_run[2][2]["conclusion"] = "skipped"
    lint_run[3][:] = files
    if accepted:
        assert control.matching_lint_run(REPO, pr)["id"] == 40
    else:
        with pytest.raises(ValueError, match="clang-tidy did not pass"):
            control.matching_lint_run(REPO, pr)


def test_comment_does_not_reset_successful_full(control, monkeypatch, pr, event):
    monkeypatch.setattr(control, "validated_request", lambda *args: pr)
    monkeypatch.setattr(control, "already_running_or_passed", lambda *args: True)
    monkeypatch.setattr(control, "api", lambda *args: pytest.fail("No writes for an already green revision"))
    control.dispatch(REPO, event)


def test_comment_with_failed_lint_never_writes(control, monkeypatch, pr, event):
    monkeypatch.setattr(control, "validated_request", lambda *args: pr)
    monkeypatch.setattr(control, "already_running_or_passed", lambda *args: False)

    def fail(*args):
        raise ValueError("PR lint job pre-commit did not pass")

    monkeypatch.setattr(control, "matching_lint_run", fail)
    monkeypatch.setattr(control, "api", lambda *args: pytest.fail("No writes before lint passes"))
    with pytest.raises(ValueError, match="did not pass"):
        control.dispatch(REPO, event)


@pytest.mark.parametrize("case", ["green", "changed-base", "failed", "rerunning", "untrusted", "cancelled"])
def test_already_green_requires_verified_full_revision(control, monkeypatch, pr, publication, case):
    run, jobs, _, _ = publication
    monkeypatch.setattr(
        control,
        "pages",
        lambda path, key: jobs
        if key == "jobs"
        else [
            {
                "context": "ci/full",
                "state": "success",
                "target_url": f"https://github.com/{REPO}/actions/runs/50/attempts/2",
            }
        ],
    )
    if case == "changed-base":

        def stale(*args):
            raise ValueError("PR is 11 commits behind main; rebase")

        monkeypatch.setattr(control, "verify_result_freshness", stale)
    elif case == "failed":
        jobs[0]["conclusion"] = "failure"
    elif case == "rerunning":
        run["run_attempt"] = 3
    elif case == "untrusted":
        run["head_branch"] = "feature"
    elif case == "cancelled":
        run["conclusion"] = "cancelled"
    assert control.already_running_or_passed(REPO, pr) == (case == "green")


@pytest.mark.parametrize("state", ["queued", "in_progress", "requested", "waiting", "pending"])
@pytest.mark.parametrize("attempt", [1, 2])
def test_consecutive_comments_honor_bound_run_before_history_indexes_it(
    control, monkeypatch, pr, event, publication, state, attempt
):
    run, _, _, _ = publication
    run.update(id=51, run_attempt=attempt, status=state, conclusion=None)
    statuses, writes, history_reads, run_reads = [], [], [], []
    monkeypatch.setattr(control, "validated_request", lambda *args: pr)
    monkeypatch.setattr(control, "matching_lint_run", lambda *args: {"id": 40})

    def api(path, payload=None):
        if payload is not None:
            writes.append((path, payload))
            if path == f"repos/{REPO}/statuses/{HEAD}":
                statuses[:] = [payload]
                return None
            if path == f"repos/{REPO}/actions/workflows/ci-full-progress.yml/dispatches":
                assert payload == {"ref": "main", "inputs": {"run_id": "51"}}
                return None
            assert path == f"repos/{REPO}/actions/workflows/ci-full.yml/dispatches"
            return {"workflow_run_id": 51}
        if path == f"repos/{REPO}/actions/runs/51":
            run_reads.append(path)
            return run
        if path == f"repos/{REPO}/actions/workflows/ci-full.yml":
            return {"id": 99}
        if path == f"repos/{REPO}/pulls/123":
            return pr
        if path == f"repos/{REPO}/git/commits/{MERGE}":
            return {"parents": [{"sha": BASE}, {"sha": HEAD}]}
        assert "/actions/runs?" in path
        history_reads.append(path)
        return {"workflow_runs": []}  # Both comments see an unindexed run list.

    def pages(path, key):
        assert path == f"repos/{REPO}/commits/{HEAD}/status" and key == "statuses"
        return statuses

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setattr(control, "pages", pages)
    control.dispatch(REPO, event)
    assert len(writes) == 4
    assert statuses[0]["target_url"].endswith("/runs/51/attempts/1")
    control.dispatch(REPO, event)
    assert len(writes) == 4, "The second comment must neither dispatch nor overwrite the binding"
    assert len(history_reads) == 1 and len(run_reads) == 1


@pytest.fixture
def bound_pending(control, monkeypatch, publication):
    run, _, _, _ = publication
    run.update(status="queued", conclusion=None)
    status = {
        "context": "ci/full",
        "state": "pending",
        "target_url": f"https://github.com/{REPO}/actions/runs/50/attempts/2",
    }

    def pages(path, key):
        assert path == f"repos/{REPO}/commits/{HEAD}/status" and key == "statuses"
        return [status]

    monkeypatch.setattr(control, "pages", pages)
    return run, status


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 51),
        ("workflow_id", 100),
        ("event", "pull_request"),
        ("head_branch", "feature"),
        ("path", ".github/workflows/other.yml"),
        ("repository", {"id": 2, "full_name": REPO}),
        ("repository", {"id": 1, "full_name": "other/repo"}),
        ("head_repository", {"id": 2}),
        ("display_title", f"CI full PR #124 @{HEAD} base:{BASE}"),
        ("display_title", f"CI full PR #123 @{'d' * 40} base:{BASE}"),
        ("display_title", f"CI full PR #123 @{HEAD} base:invalid"),
        ("run_attempt", 1),
        ("run_attempt", "2"),
    ],
)
def test_bound_pending_requires_matching_trusted_revision(control, pr, bound_pending, field, value):
    bound_pending[0][field] = value
    assert not control.already_running_or_passed(REPO, pr)


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "success"])
def test_completed_pending_binding_does_not_block_retry(control, pr, bound_pending, conclusion):
    bound_pending[0].update(status="completed", conclusion=conclusion)
    assert not control.already_running_or_passed(REPO, pr)


def test_unavailable_bound_run_never_dispatches(control, monkeypatch, pr, event, bound_pending):
    monkeypatch.setattr(control, "validated_request", lambda *args: pr)

    def unavailable(path, payload=None):
        assert payload is None, "An unverifiable bound run must not dispatch or overwrite its status"
        assert path == f"repos/{REPO}/actions/runs/50"
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "api", unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        control.dispatch(REPO, event)


def test_downstream_gate_cannot_be_skipped(control, needs):
    needs["lint-gate"]["result"] = "skipped"
    with pytest.raises(ValueError, match="lint-gate"):
        control.aggregate(needs)


def test_full_workflow_reuses_lint_without_running_it():
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci-full.yml").read_text())["jobs"]
    assert "reuse_lint != 'true'" in jobs["pre-commit"]["if"]
    assert "reuse_lint != 'true'" in jobs["clang-tidy"]["if"]
    for name in ("unit-tests", "system-tests", "system-tests-direct", "prebuild-wheels-github"):
        assert "lint-gate" in jobs[name]["needs"]
        assert "pre-commit" not in jobs[name]["needs"]
        assert "clang-tidy" not in jobs[name]["needs"]


@pytest.mark.parametrize(
    "reuse,docs,pre,clang,accepted",
    [
        ("true", "false", "skipped", "skipped", True),
        ("true", "false", "success", "skipped", False),
        ("true", "false", "skipped", "failure", False),
        ("false", "false", "success", "success", True),
        ("false", "true", "success", "skipped", True),
        ("false", "false", "success", "skipped", False),
        ("false", "true", "failure", "skipped", False),
        ("false", "false", "success", "failure", False),
    ],
)
def test_lint_gate_shell_rejects_unjustified_skips(reuse, docs, pre, clang, accepted):
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci-full.yml").read_text())["jobs"]
    result = subprocess.run(
        ["bash", "-e", "-c", jobs["lint-gate"]["steps"][0]["run"]],
        check=False,
        env={**os.environ, "REUSE_LINT": reuse, "DOCS_ONLY": docs, "PRE_COMMIT": pre, "CLANG_TIDY": clang},
        capture_output=True,
    )
    assert (result.returncode == 0) == accepted


def test_lint_reuse_waits_for_rerun_attempt(control, pr, lint_run):
    lint_run[0]["status"] = "in_progress"
    with pytest.raises(ValueError, match="still running"):
        control.matching_lint_run(REPO, pr)


def test_effective_jobs_preserves_successful_dependencies(control, monkeypatch):
    attempts = {
        1: [
            {"name": "pre-commit", "id": 1, "conclusion": "success"},
            {"name": "clang-tidy", "id": 2, "conclusion": "failure"},
        ],
        2: [{"name": "clang-tidy", "id": 3, "conclusion": "success"}],
    }

    def pages(path, key):
        assert key == "jobs"
        number = int(path.split("/attempts/")[1].split("/")[0])
        return attempts[number]

    monkeypatch.setattr(control, "pages", pages)
    jobs = control.effective_jobs(REPO, {"id": 40, "run_attempt": 2})
    assert {job["name"]: job["id"] for job in jobs} == {"pre-commit": 1, "clang-tidy": 3}


def test_every_downstream_job_explicitly_tolerates_skipped_lint_ancestors():
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci-full.yml").read_text())["jobs"]
    for name, job in jobs.items():
        if name == "full-result" or "lint-gate" not in job.get("needs", []):
            continue
        condition = job["if"]
        assert "!cancelled()" in condition, name
        assert "needs.lint-gate.result == 'success'" in condition, name
        assert "needs.changes.result == 'success'" in condition, name
        if "toolchain" in job["needs"]:
            assert "needs.toolchain.result == 'success'" in condition, name


@pytest.mark.parametrize("title", [None, "", "CI", f"CI full PR #456 @{'d' * 40} base:{BASE}"])
def test_router_resolves_webhook_title_from_exact_api_attempt(
    control, monkeypatch, publication, publication_event, tmp_path, title
):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    source = publication_event["workflow_run"]
    actual = copy.deepcopy(source)
    if title is None:
        source.pop("display_title")
    else:
        source["display_title"] = title

    def api(path):
        assert path == f"repos/{REPO}/actions/runs/50/attempts/1"
        return actual

    monkeypatch.setattr(control, "api", api)
    control.route(REPO, publication_event)
    outputs = dict(line.split("=", 1) for line in target.read_text().splitlines())
    assert outputs["key"] == f"123-{HEAD}"
    routed = json.loads(outputs["event"])
    assert routed["workflow_run"]["display_title"] == actual["display_title"]
    assert control.verified_event_run(REPO, routed)["run_attempt"] == 1


@pytest.mark.parametrize("field", ["id", "workflow_id", "run_number", "run_attempt"])
def test_router_rejects_api_identity_mismatch(control, monkeypatch, publication_event, tmp_path, field):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    actual = copy.deepcopy(publication_event["workflow_run"])
    actual[field] += 1
    monkeypatch.setattr(control, "api", lambda path: actual)
    with pytest.raises(control.UntrustedRunError):
        control.route(REPO, publication_event)
    assert not target.exists()


def test_router_ignores_manual_main_ci_without_pr_inputs(control, monkeypatch, publication_event, tmp_path):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    actual = {**publication_event["workflow_run"], "display_title": "CI workflow_dispatch main"}
    monkeypatch.setattr(control, "api", lambda path: actual)
    control.route(REPO, publication_event)
    assert not target.exists()


def test_reconciliation_publishes_existing_result_without_dispatching(
    control, monkeypatch, publication, tmp_path
):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    event = {"inputs": {"run_id": "50"}}
    control.route(REPO, event)
    outputs = dict(line.split("=", 1) for line in target.read_text().splitlines())
    assert outputs["key"] == f"123-{HEAD}"
    monkeypatch.setenv("CI_ROUTED_EVENT", outputs["event"])
    control.publish(REPO, event)
    assert len(publication[3]) == 1
    path, payload = publication[3][0]
    assert path == f"repos/{REPO}/statuses/{HEAD}"
    assert payload["state"] == "success"
    assert payload["target_url"].endswith("/runs/50/attempts/2")


@pytest.mark.parametrize("ref", ["refs/heads/feature", "refs/tags/main", ""])
def test_reconciliation_requires_main(control, monkeypatch, ref):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", ref)
    with pytest.raises(control.UntrustedRunError, match="must run from main"):
        control.route(REPO, {"inputs": {"run_id": "50"}})


@pytest.mark.parametrize("run_id", ["", "0", "-1", "1/attempts/2", "1\n", 50])
def test_reconciliation_rejects_invalid_run_id(control, monkeypatch, run_id):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    with pytest.raises(ValueError, match="positive workflow run ID"):
        control.route(REPO, {"inputs": {"run_id": run_id}})


def test_routed_identity_invalidates_result_on_api_failure(control, monkeypatch, publication, tmp_path):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    event = {"inputs": {"run_id": "50"}}
    control.route(REPO, event)
    outputs = dict(line.split("=", 1) for line in target.read_text().splitlines())
    monkeypatch.setenv("CI_ROUTED_EVENT", outputs["event"])

    def unavailable(path, payload=None):
        if payload is not None:
            publication[3].append((path, payload))
            return None
        assert path.endswith("/actions/runs/50")
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "api", unavailable)
    control.publish(REPO, event)
    assert publication[3][0][1]["state"] == "failure"


@pytest.mark.parametrize("action", ["in_progress", "completed"])
def test_failed_route_reaches_failure_only_publisher(
    control, monkeypatch, publication, publication_event, tmp_path, action
):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    publication_event["action"] = action
    publication_event["workflow_run"]["run_attempt"] = 2
    reads = []

    def api(path, payload=None):
        if payload is not None:
            publication[3].append((path, payload))
            return None
        reads.append(path)
        assert path == f"repos/{REPO}/actions/runs/50/attempts/2"
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "api", api)
    control.route(REPO, publication_event)
    outputs = dict(line.split("=", 1) for line in target.read_text().splitlines())
    assert outputs["key"] == f"123-{HEAD}"
    assert json.loads(outputs["event"])["routing_failed"] is True
    monkeypatch.setenv("CI_ROUTED_EVENT", outputs["event"])
    control.publish(REPO, publication_event)
    assert len(reads) == 1, "A failed route cannot be upgraded to success by a later lookup"
    assert len(publication[3]) == 1
    path, payload = publication[3][0]
    assert path == f"repos/{REPO}/statuses/{HEAD}"
    assert payload["state"] == "failure"
    assert payload["target_url"].endswith("/runs/50/attempts/2")


@pytest.mark.parametrize("newer", ["run", "attempt", "unbound"])
def test_failed_route_preserves_newer_status_owner(
    control, monkeypatch, publication, publication_event, tmp_path, newer
):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    candidate = copy.deepcopy(publication[0])
    if newer == "run":
        candidate.update(id=51, run_number=9)
    status = {
        "context": "ci/full",
        "state": "success",
        "target_url": f"https://github.com/{REPO}/actions/runs/{candidate['id']}/attempts/2",
    }
    if newer == "unbound":
        status.update(
            state="pending",
            description="Full CI requested; waiting for dispatch",
            target_url=f"https://github.com/{REPO}/pull/123",
        )

    def api(path, payload=None):
        assert payload is None, "A delayed failure-only route must preserve the newer writer"
        if path == f"repos/{REPO}/actions/runs/50/attempts/1":
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        expected = "51" if newer == "run" else "50/attempts/2"
        assert path == f"repos/{REPO}/actions/runs/{expected}"
        return candidate

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setattr(control, "pages", lambda *args: [status])
    control.route(REPO, publication_event)
    outputs = dict(line.split("=", 1) for line in target.read_text().splitlines())
    monkeypatch.setenv("CI_ROUTED_EVENT", outputs["event"])
    control.publish(REPO, publication_event)


@pytest.mark.parametrize("title", [None, "", "CI"])
def test_failed_route_cannot_invent_pr_identity(control, monkeypatch, publication_event, tmp_path, title):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    if title is None:
        publication_event["workflow_run"].pop("display_title")
    else:
        publication_event["workflow_run"]["display_title"] = title

    def unavailable(path):
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "api", unavailable)
    with pytest.raises(control.UntrustedRunError):
        control.route(REPO, publication_event)
    assert not target.exists()


def test_progress_resolution_keeps_api_failure_for_retry(control, monkeypatch, publication_event):
    def unavailable(path):
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(control, "api", unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        control.resolved_event(REPO, publication_event)


def test_full_finalizer_validates_all_main_dispatches_and_only_notifies_for_prs():
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci-full.yml").read_text())["jobs"]
    finalizer = jobs["full-result"]
    assert finalizer["runs-on"] == "ubuntu-latest"
    assert finalizer["permissions"]["actions"] == "write"
    assert finalizer["if"] == (
        "always() && github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'"
    )
    aggregate = next(
        step for step in finalizer["steps"] if step.get("run", "").endswith("ci_control.py aggregate")
    )
    assert "if" not in aggregate
    assert aggregate["env"]["CI_NEEDS"] == "${{ toJSON(needs) }}"
    notifications = finalizer["steps"][-2:]
    for step, workflow in zip(notifications, ["ci-full-status.yml", "ci-full-progress.yml"]):
        assert step["if"] == "always() && inputs.pr_number != ''"
        assert f"gh workflow run {workflow}" in step["run"]
        assert "--ref main" in step["run"]
        assert 'run_id="$GITHUB_RUN_ID"' in step["run"]
    assert "wait_for_completion=true" in notifications[0]["run"]
    assert notifications[1]["continue-on-error"] is True
    status = yaml.safe_load((ROOT / ".github/workflows/ci-full-status.yml").read_text())
    route_steps = status["jobs"]["route"]["steps"]
    assert route_steps[-2]["run"].endswith("ci_control.py route")
    assert "inputs.wait_for_completion" in route_steps[-1]["if"]
    assert "concurrency" not in status["jobs"]["route"]
    progress = yaml.safe_load((ROOT / ".github/workflows/ci-full-progress.yml").read_text())
    assert progress[True]["workflow_dispatch"]["inputs"]["run_id"]["required"]
    reconcile = progress["jobs"]["progress"]["steps"][-1]
    assert reconcile["if"] == "always() && steps.monitor.outputs.source_run != ''"
    assert reconcile["env"]["SOURCE_RUN"] == "${{ steps.monitor.outputs.source_run }}"
    assert "gh workflow run ci-full-status.yml" in reconcile["run"]


@pytest.mark.parametrize("recovers", [False, True])
def test_status_completion_poll_retries_api_errors_without_unbounded_wait(tmp_path, monkeypatch, recovers):
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci-full-status.yml").read_text())
    script = workflow["jobs"]["route"]["steps"][-1]["run"]
    counter = tmp_path / "polls"
    counter.write_text("0")
    monkeypatch.setenv("CI_TEST_POLL_COUNT", str(counter))
    monkeypatch.setenv("CI_TEST_RECOVERS", str(int(recovers)))
    # Execute the real workflow shell with Actions' errexit behavior. The
    # command substitution runs in a subshell, so keep its counter in a file.
    fake_api = """
gh() {
  count=$(cat "$CI_TEST_POLL_COUNT")
  count=$((count + 1))
  echo "$count" > "$CI_TEST_POLL_COUNT"
  if [ "$CI_TEST_RECOVERS" = 1 ] && [ "$count" -ge 3 ]; then
    echo completed
  elif [ "$CI_TEST_RECOVERS" = 1 ] && [ "$count" = 2 ]; then
    echo in_progress
  else
    return 1
  fi
}
sleep() { :; }
"""
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", fake_api + script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == (0 if recovers else 1), result.stderr
    assert int(counter.read_text()) == (3 if recovers else 60)


def test_split_workflows_create_only_lint_jobs_for_pull_requests(control):
    lint = yaml.safe_load((ROOT / control.LINT_WORKFLOW).read_text())
    full = yaml.safe_load((ROOT / control.WORKFLOW).read_text())
    assert lint["name"] == "CI"  # Preserve required contexts and pre-rollout lint lookup.
    assert set(lint[True]) == {"pull_request"}
    assert set(lint["jobs"]) == {"changes", "toolchain", "pre-commit", "clang-tidy", "lint-gate"}
    assert full["name"] == "CI Full"
    assert set(full[True]) == {"push", "workflow_dispatch"}
    assert "pull_request" not in full["run-name"]
    for name, job in lint["jobs"].items():
        assert job == full["jobs"][name], name
    for filename in ("ci-full-progress.yml", "ci-full-status.yml"):
        publisher = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
        assert publisher[True]["workflow_run"]["workflows"] == ["CI", "CI Full"]


def test_generated_lint_workflow_is_current_and_excludes_new_full_jobs(control):
    spec = importlib.util.spec_from_file_location(
        "ci_lint_generator", ROOT / "tests/lint/generate_ci_lint.py"
    )
    assert spec and spec.loader
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    source = (ROOT / control.WORKFLOW).read_text()
    expected = (ROOT / control.LINT_WORKFLOW).read_text()
    assert generator.render(source) == expected
    # A new heavy job in the source must not silently appear on every PR.
    assert generator.render(source + "\n  another-full-test:\n    runs-on: ubuntu-latest\n") == expected


@pytest.mark.parametrize("legacy", [False, True])
def test_publish_and_manual_recovery_accept_both_full_workflow_generations(
    control, monkeypatch, publication, legacy
):
    run, _, _, writes = publication
    path = control.LINT_WORKFLOW if legacy else control.WORKFLOW
    run["path"] = path
    original = control.api

    def api(endpoint, payload=None):
        if endpoint.endswith("/workflows/" + Path(path).name):
            return {"id": 99, "path": path}
        return original(endpoint, payload)

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    routed = control.resolved_event(REPO, {"inputs": {"run_id": "50"}})
    assert routed["workflow_run"]["path"] == path
    control.publish(REPO, routed)
    assert writes[-1][1]["state"] == "success"


@pytest.mark.parametrize("source_legacy", [False, True])
@pytest.mark.parametrize("candidate_newer", [False, True])
def test_cross_workflow_requests_use_creation_time_not_workflow_sequence(
    control, monkeypatch, publication, source_legacy, candidate_newer
):
    run = publication[0]
    run["path"] = control.LINT_WORKFLOW if source_legacy else control.WORKFLOW
    # A new workflow can have run_number=1 while an old workflow is at 1000.
    run["run_number"] = 1000 if source_legacy else 1
    candidate: dict[str, Any] = {
        **run,
        "path": control.WORKFLOW if source_legacy else control.LINT_WORKFLOW,
        "workflow_id": 100,
        "id": 51 if candidate_newer else 49,
        "run_number": 1 if source_legacy else 1000,
        "created_at": "2026-11-05T01:00:01Z" if candidate_newer else "2026-11-05T00:59:59Z",
    }

    def api(path, payload=None):
        assert payload is None
        if "/actions/runs?" in path:
            return {"workflow_runs": [candidate]}
        if path.endswith(f"/runs/{candidate['id']}"):
            return candidate
        assert path.endswith("/workflows/" + Path(candidate["path"]).name)
        return {"id": 100, "path": candidate["path"]}

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {
                "context": "ci/full",
                "state": "pending",
                "target_url": f"https://github.com/{REPO}/actions/runs/{candidate['id']}/attempts/1",
            }
        ],
    )
    assert control.newer_request_exists(REPO, 123, HEAD, run) == candidate_newer
    assert control.newer_status_exists(REPO, 123, HEAD, run) == candidate_newer


@pytest.mark.parametrize("lookup_failed", [False, True])
def test_routing_retains_creation_time_for_cross_workflow_failure_fallback(
    control, monkeypatch, publication, publication_event, lookup_failed
):
    source = publication_event["workflow_run"]
    source["path"] = control.LINT_WORKFLOW
    publication_event["workflow"]["path"] = control.LINT_WORKFLOW

    def api(path, payload=None):
        assert payload is None
        if path.endswith("/runs/50/attempts/1"):
            if lookup_failed:
                raise subprocess.CalledProcessError(1, ["gh", "api"])
            return source
        if path.endswith("/runs/51"):
            return {
                **source,
                "id": 51,
                "path": control.WORKFLOW,
                "workflow_id": 100,
                "run_number": 1,
                "created_at": "2026-11-05T01:00:01Z",
            }
        assert path.endswith("/workflows/ci-full.yml")
        return {"id": 100, "path": control.WORKFLOW}

    monkeypatch.setattr(control, "api", api)
    routed = control.resolved_event(REPO, publication_event, allow_failure_route=True)
    assert routed["workflow_run"]["created_at"] == source["created_at"]
    monkeypatch.setattr(
        control,
        "pages",
        lambda *args: [
            {
                "context": "ci/full",
                "state": "pending",
                "target_url": f"https://github.com/{REPO}/actions/runs/51/attempts/1",
            }
        ],
    )
    assert control.newer_status_exists(REPO, 123, HEAD, routed["workflow_run"])


@pytest.mark.parametrize("path", [".github/workflows/ci.yml", ".github/workflows/ci-full.yml"])
def test_full_workflow_requires_matching_path_and_id(control, monkeypatch, path):
    monkeypatch.setattr(control, "api", lambda *args: {"id": 99, "path": path})
    with pytest.raises(control.UntrustedRunError):
        control.full_workflow(REPO, {"path": path, "workflow_id": 100})


@pytest.mark.parametrize("suffix", ["", " request:900-1"])
@pytest.mark.parametrize(
    "path", [".github/workflows/ci.yml", ".github/workflows/ci-full.yml", ".github/workflows/other.yml"]
)
def test_duplicate_scan_includes_legacy_and_current_full_only(control, monkeypatch, path, suffix):
    title = f"CI full PR #123 @{HEAD} base:{BASE}"
    monkeypatch.setattr(
        control,
        "api",
        lambda *args: {
            "workflow_runs": [
                {
                    "display_title": title + suffix,
                    "head_branch": "main",
                    "event": "workflow_dispatch",
                    "path": path,
                    "status": "in_progress",
                }
            ]
        },
    )
    point = datetime(2026, 11, 5, tzinfo=timezone.utc)
    assert control.unfinished_request_exists(REPO, title, point, point) == (path in control.FULL_WORKFLOWS)


@pytest.fixture
def a5_control(monkeypatch, controller):
    monkeypatch.setitem(sys.modules, "ci_control", controller)
    spec = importlib.util.spec_from_file_location(
        "ci_a5_distributed", ROOT / ".github/scripts/ci_a5_distributed.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("GITHUB_RUN_ID", "1234")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    return module


@pytest.fixture
def a5_request(a5_control, monkeypatch, pr, event):
    event["comment"]["body"] = a5_control.COMMAND
    writes, outputs = [], {}

    def api(path, payload=None, **kwargs):
        if payload is not None:
            writes.append((path, payload, kwargs))
            return {"id": 987}
        if path == f"repos/{REPO}/pulls/123":
            return copy.deepcopy(pr)
        if path == f"repos/{REPO}/git/ref/heads/main":
            return {"ref": "refs/heads/main", "object": {"type": "commit", "sha": BASE}}
        if path == f"repos/{REPO}/git/commits/{MERGE}":
            return {"parents": [{"sha": BASE}, {"sha": HEAD}]}
        pytest.fail(f"Unexpected API: {path}")

    monkeypatch.setattr(a5_control, "api", api)
    monkeypatch.setattr(a5_control, "head_checks", lambda *_: [])
    monkeypatch.setattr(a5_control, "output", outputs.update)
    monkeypatch.setattr(a5_control, "matching_lint_run", lambda *_: {"conclusion": "success"})
    # The shared main-branch validators call the controller API directly.
    monkeypatch.setattr(sys.modules["ci_control"], "api", api)
    return pr, event, writes, outputs


def test_a5_command_pins_merge_and_creates_independent_check(a5_control, a5_request):
    _, event, writes, outputs = a5_request
    # External/fork commenters have the same admission policy as /ci full.
    event["comment"]["body"] = "  /ci a5-distributed\n"
    a5_control.prepare(REPO, event)
    assert outputs == {"checkout_sha": MERGE, "check_id": "987"}
    assert len(writes) == 1
    path, payload, _ = writes[0]
    assert path == f"repos/{REPO}/check-runs"
    assert payload["name"] == "ci/a5-distributed"
    assert payload["head_sha"] == HEAD
    assert payload["status"] == "in_progress"
    assert payload["details_url"].endswith("/runs/1234/attempts/1")


@pytest.mark.parametrize("status", ["missing", "in_progress", "failure", "cancelled"])
def test_a5_requires_passing_lint_before_publishing_outputs(a5_control, a5_request, monkeypatch, status):
    pr, event, writes, outputs = a5_request

    def lint(repo, candidate, command):
        assert repo == REPO and candidate == pr and command == "/ci a5-distributed"
        if status in {"missing", "in_progress"}:
            raise ValueError(f"Lint {status} before {command}")
        return {"conclusion": status}

    monkeypatch.setattr(a5_control, "matching_lint_run", lint)
    with pytest.raises(ValueError, match="/ci a5-distributed"):
        a5_control.prepare(REPO, event)
    assert not writes and not outputs


@pytest.mark.parametrize("case", ["suffix", "full", "edited", "issue"])
def test_a5_ignores_unrelated_comments(a5_control, a5_request, monkeypatch, case):
    _, event, writes, outputs = a5_request
    if case == "suffix":
        event["comment"]["body"] += " extra"
    elif case == "full":
        event["comment"]["body"] = "/ci full"
    elif case == "edited":
        event["action"] = "edited"
    else:
        del event["issue"]["pull_request"]
    monkeypatch.setattr(a5_control, "api", lambda *a: pytest.fail("Ignored comments must not read or write"))
    a5_control.prepare(REPO, event)
    assert not writes and not outputs


@pytest.mark.parametrize("case", ["integration", "closed", "repo", "deleted-fork", "conflict", "sha"])
def test_a5_rejects_invalid_pr_before_writes(a5_control, a5_request, case):
    pr, event, writes, outputs = a5_request
    if case == "integration":
        pr["base"]["ref"] = "integration/urma-ccu-test"
    elif case == "closed":
        pr["state"] = "closed"
    elif case == "repo":
        pr["base"]["repo"]["full_name"] = "different/repo"
    elif case == "deleted-fork":
        pr["head"]["repo"] = None
    elif case == "conflict":
        pr["mergeable"] = False
    else:
        pr["head"]["sha"] = "invalid"
    with pytest.raises(ValueError):
        a5_control.prepare(REPO, event)
    assert not writes and not outputs


@pytest.mark.parametrize(
    "case", ["recovers", "head-changed", "base-changed", "during-lookup", "timeout", "api-failure"]
)
def test_a5_waits_only_for_consistent_merge(a5_control, a5_request, monkeypatch, case):
    pr, event, writes, outputs = a5_request
    api = a5_control.api
    elapsed = [0]
    monkeypatch.setattr(
        a5_control,
        "time",
        SimpleNamespace(
            monotonic=lambda: elapsed[0], sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)
        ),
    )

    def delayed_api(path, payload=None, **kwargs):
        result = api(path, payload, **kwargs)
        if payload is not None:
            return result
        if path.endswith(f"/git/commits/{MERGE}"):
            if case == "api-failure":
                raise RuntimeError("API unavailable")
            if case == "during-lookup":
                pr["head"]["sha"] = "e" * 40
            elif not elapsed[0] or case == "timeout":
                result["parents"][0]["sha"] = "d" * 40
        if elapsed[0] and case == "head-changed" and path.endswith("/pulls/123"):
            result["head"]["sha"] = "e" * 40
        if elapsed[0] and case == "base-changed" and "/git/ref/" in path:
            result["object"]["sha"] = "f" * 40
        return result

    monkeypatch.setattr(sys.modules["ci_control"], "api", delayed_api)
    monkeypatch.setattr(a5_control, "api", delayed_api)
    if case == "recovers":
        a5_control.prepare(REPO, event)
        assert elapsed[0] == 5
        assert outputs["checkout_sha"] == MERGE
    else:
        # Pin changes abort at the first observation. Only a stale merge waits until timeout.
        error, message, seconds = {
            "head-changed": (ValueError, "head or base changed", 5),
            "base-changed": (ValueError, "head or base changed", 5),
            "during-lookup": (ValueError, "changed during validation", 0),
            "timeout": (ValueError, "test merge is not ready", 60),
            "api-failure": (RuntimeError, "API unavailable", 0),
        }[case]
        with pytest.raises(error, match=message):
            a5_control.prepare(REPO, event)
        assert not writes and not outputs
        assert elapsed[0] == seconds


@pytest.mark.parametrize("result", ["success", "failure", "skipped", "cancelled"])
@pytest.mark.parametrize("job", ["changes", "toolchain", "onboard-tests-a5-distributed"])
def test_a5_result_requires_every_dependency(a5_control, monkeypatch, result, job):
    needs: dict[str, Any] = {name: {"result": "success"} for name in ("changes", "toolchain", a5_control.JOB)}
    needs["changes"]["outputs"] = {"check_id": "987"}
    needs[job]["result"] = result
    writes = []
    monkeypatch.setattr(a5_control, "api", lambda *args, **kwargs: writes.append((args, kwargs)))
    if result == "success":
        a5_control.finish(REPO, needs)
    else:
        with pytest.raises(ValueError, match="did not pass"):
            a5_control.finish(REPO, needs)
    (path, payload), kwargs = writes[0]
    assert path == f"repos/{REPO}/check-runs/987"
    assert kwargs == {"method": "PATCH"}
    assert payload["details_url"].endswith("/attempts/1")
    assert payload["status"] == "completed"
    assert payload["conclusion"] == ("success" if result == "success" else "failure")


def test_a5_output_failure_closes_orphan_check(a5_control, a5_request, monkeypatch):
    _, event, writes, _ = a5_request

    def broken_output(_):
        raise OSError("outputs unavailable")

    monkeypatch.setattr(a5_control, "output", broken_output)
    with pytest.raises(OSError, match="outputs unavailable"):
        a5_control.prepare(REPO, event)
    path, payload, options = writes[-1]
    assert path.endswith("/check-runs/987") and options == {"method": "PATCH"}
    assert payload["conclusion"] == "failure"


@pytest.mark.parametrize("owner_key", ["1234", "a5-pr-123:1234"])
@pytest.mark.parametrize("output_fails", [False, True])
def test_a5_full_rerun_admission_publishes_progress(
    a5_control, a5_request, monkeypatch, output_fails, owner_key
):
    _, event, writes, outputs = a5_request
    original = {
        "name": "ci/a5-distributed",
        "id": 41,
        "external_id": owner_key,
        "app": {"slug": "github-actions", "id": 42},
        "head_sha": HEAD,
        "status": "completed",
        "output": {"summary": None},
    }
    api = a5_control.api

    def publish(values):
        # prepare must create the current check before downstream jobs get the original ID.
        assert len(writes) == 1 and writes[0][1]["status"] == "in_progress"
        if output_fails:
            raise OSError("outputs unavailable")
        outputs.update(values)

    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setattr(a5_control, "head_checks", lambda *_: [original])
    monkeypatch.setattr(
        a5_control, "api", lambda path, *a, **k: original if path.endswith("/41") else api(path, *a, **k)
    )
    monkeypatch.setattr(a5_control, "output", publish)
    if output_fails:
        with pytest.raises(OSError, match="outputs unavailable"):
            a5_control.prepare(REPO, event)
        path, payload, options = writes[1]
        assert path.endswith("/check-runs/987") and options == {"method": "PATCH"}
        assert payload["conclusion"] == "failure"
    else:
        a5_control.prepare(REPO, event)
        assert outputs == {"checkout_sha": MERGE, "check_id": "41"}
    assert writes[0][1]["external_id"] == f"{owner_key}:2:41"
    assert writes[0][1]["details_url"].endswith("/attempts/2")


@pytest.mark.parametrize("state", ["queued", "in_progress"])
def test_a5_observer_yields_to_full_rerun_admission(a5_control, monkeypatch, state):
    run = {
        "id": 1234,
        "path": ".github/workflows/ci-a5-distributed.yml",
        "event": "issue_comment",
        "head_branch": "main",
        "display_title": "A5 distributed PR #123 (comment 17)",
        "run_attempt": 2,
        "status": "in_progress",
    }
    monkeypatch.setattr(a5_control, "api", lambda path: run)
    monkeypatch.setattr(a5_control, "pages", lambda *_: [{"name": "changes", "status": state}])
    monkeypatch.setattr(a5_control.time, "sleep", lambda _: pytest.fail("observer held the admission lock"))
    a5_control.route_rerun(REPO, run)
    a5_control.reopen(REPO, run)


@pytest.mark.parametrize("state", ["queued", "in_progress", "completed"])
@pytest.mark.parametrize("carried_device", [False, True])
def test_a5_result_retry_does_not_request_the_writer_lock(a5_control, monkeypatch, state, carried_device):
    run = {
        "id": 1234,
        "path": ".github/workflows/ci-a5-distributed.yml",
        "event": "issue_comment",
        "head_branch": "main",
        "display_title": "A5 distributed PR #123 (comment 99)",
        "run_attempt": 2,
        "status": "in_progress",
    }
    jobs = [{"name": "result", "run_attempt": 2, "status": state}]
    if carried_device:
        jobs.append(
            {"name": "onboard-tests-a5-distributed / check 41", "run_attempt": 2, "status": "completed"}
        )
    monkeypatch.setattr(a5_control, "api", lambda path: run)
    monkeypatch.setattr(a5_control, "pages", lambda *_: jobs)
    monkeypatch.setattr(a5_control.time, "sleep", lambda _: pytest.fail("waited for result"))
    monkeypatch.setattr(a5_control, "output", lambda _: pytest.fail("requested the writer lock"))
    a5_control.route_rerun(REPO, run)


@pytest.mark.parametrize("status, attempts", [(503, 3), (403, 1)])
def test_a5_check_update_has_bounded_retries(a5_control, monkeypatch, status, attempts):
    calls = []

    def unavailable(path, payload, **options):
        calls.append((path, payload, options))
        raise subprocess.CalledProcessError(1, "gh", stderr=f"request failed (HTTP {status})")

    monkeypatch.setattr(a5_control, "api", unavailable)
    monkeypatch.setattr(a5_control.time, "sleep", lambda _: None)
    with pytest.raises(subprocess.CalledProcessError):
        a5_control.update_check(REPO, "123", {"status": "completed"})
    assert (
        calls == [(f"repos/{REPO}/check-runs/123", {"status": "completed"}, {"method": "PATCH"})] * attempts
    )


def test_a5_device_rerun_refreshes_previous_attempt(a5_control, monkeypatch):
    run = {
        "id": 1234,
        "run_attempt": 2,
        "event": "issue_comment",
        "head_branch": "main",
        "display_title": "A5 distributed PR #123 (comment 17)",
        "path": ".github/workflows/ci-a5-distributed.yml",
        "status": "in_progress",
    }
    original = {
        "id": 987,
        "name": "ci/a5-distributed",
        "external_id": "1234",
        "status": "completed",
        "output": {"summary": None},
        "head_sha": HEAD,
        "app": {"id": 42},
    }
    created = []

    def api(path, payload=None, **kwargs):
        if payload:
            assert path.endswith("/check-runs")
            created.append({**payload, "id": 988, "app": {"id": 42}})
            return created[-1]
        if path.endswith("/runs/1234"):
            return run
        assert path.endswith("/check-runs/987")
        return original

    monkeypatch.setattr(a5_control, "api", api)
    monkeypatch.setattr(
        a5_control,
        "pages",
        lambda _, key: [
            {"name": "onboard-tests-a5-distributed / check 987", "run_attempt": 2, "status": "queued"}
        ]
        if key == "jobs"
        else created,
    )
    routed = {}
    monkeypatch.setattr(a5_control, "output", routed.update)
    a5_control.route_rerun(REPO, run)
    assert routed == {"number": "123"}
    a5_control.reopen(REPO, run)
    assert len(created) == 1 and created[0]["status"] == "in_progress"
    assert created[0]["details_url"].endswith("/runs/1234/attempts/2")
    assert created[0]["details_url"] in created[0]["output"]["summary"]
    # A late observer or finalizer reuses the new attempt check, even after it completed.
    created[0]["status"] = "completed"
    assert a5_control.attempt_check(REPO, "987", "1234", "2") == "988"
    assert len(created) == 1


@pytest.mark.parametrize(
    "state,conclusion,pr_number,reuse",
    [
        ("queued", None, 123, True),
        ("in_progress", None, 123, True),
        ("completed", "success", 123, True),
        ("completed", "failure", 123, False),
        ("completed", "cancelled", 123, False),
        ("in_progress", None, 999, False),
    ],
)
def test_a5_duplicate_command_admission(
    a5_control, a5_request, monkeypatch, state, conclusion, pr_number, reuse
):
    _, event, writes, outputs = a5_request
    url = a5_control.attempt_url(REPO, "44", "2")
    check = {
        "id": 88,
        "status": "completed" if conclusion else "in_progress",
        "external_id": "44:2:33",
        "app": {"slug": "github-actions"},
        "conclusion": conclusion,
        **a5_control.attempt_fields(url, "Device result"),
    }
    run = {
        "path": ".github/workflows/ci-a5-distributed.yml",
        "event": "issue_comment",
        "head_branch": "main",
        "display_title": f"A5 distributed PR #{pr_number} (comment 12)",
        "run_attempt": 2,
        "status": state,
        "conclusion": conclusion,
    }
    api = a5_control.api
    monkeypatch.setattr(a5_control, "head_checks", lambda repo, head: [check] if head == HEAD else [])
    monkeypatch.setattr(
        a5_control, "api", lambda path, *args: run if path.endswith("/actions/runs/44") else api(path, *args)
    )
    a5_control.prepare(REPO, event)
    assert bool(writes) == bool(outputs) == (not reuse)


@pytest.mark.parametrize(
    "conclusion",
    [
        "failure",
        "cancelled",
        "timed_out",
        "stale",
        "action_required",
        "neutral",
        "skipped",
        "startup_failure",
    ],
)
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_a5_fresh_command_retires_interrupted_publisher(
    a5_control, a5_request, monkeypatch, conclusion, cleanup_fails
):
    _, event, writes, outputs = a5_request
    stale = {"id": 88, "external_id": "44", "app": {"slug": "github-actions"}, "status": "in_progress"}
    checks = [stale, {**stale, "id": 89, "external_id": "44:2:88", "status": "completed"}]
    run = {
        "path": ".github/workflows/ci-a5-distributed.yml",
        "event": "issue_comment",
        "head_branch": "main",
        "display_title": "A5 distributed PR #123 (comment 12)",
        "run_attempt": 2,
        "status": "completed",
        "conclusion": conclusion,
    }
    api = a5_control.api

    def retry_api(path, *args, **kwargs):
        if path.endswith("/actions/runs/44"):
            return run
        if cleanup_fails and path.endswith("/check-runs/88"):
            raise ValueError("Cleanup unavailable")
        return api(path, *args, **kwargs)

    monkeypatch.setattr(a5_control, "api", retry_api)
    monkeypatch.setattr(a5_control, "head_checks", lambda repo, head: checks if head == HEAD else [])
    if cleanup_fails:
        with pytest.raises(ValueError, match="Cleanup unavailable"):
            a5_control.prepare(REPO, event)
        assert not writes and not outputs
    else:
        a5_control.prepare(REPO, event)
        assert writes[0] == (
            f"repos/{REPO}/check-runs/88",
            {"status": "completed", "conclusion": "cancelled"},
            {"method": "PATCH"},
        )
        assert len(writes) == 2 and writes[1][0] == f"repos/{REPO}/check-runs"
        assert outputs == {"checkout_sha": MERGE, "check_id": "987"}


@pytest.mark.parametrize("unfinished", ["original", "rerun", "both"])
def test_a5_retry_finishes_pending_check(a5_control, monkeypatch, unfinished):
    original: dict[str, Any] = {
        "id": 41,
        "external_id": "1234",
        "name": "ci/a5-distributed",
        "head_sha": HEAD,
        "app": {"id": 42},
        "output": {"summary": None},
        "status": "completed" if unfinished == "rerun" else "in_progress",
    }
    earlier: dict[str, Any] = {**original, "id": 42, "external_id": "1234:2:41", "status": "in_progress"}
    checks = [original] if unfinished == "original" else [earlier, original]
    pending = checks[0]

    def api(path, payload=None, *, method=None):
        if payload is None:
            assert path.endswith("/check-runs/41")
            return original
        assert method == "PATCH"
        target = next(c for c in checks if path.endswith(f"/check-runs/{c['id']}"))
        target.update(payload)

    monkeypatch.setattr(a5_control, "api", api)
    monkeypatch.setattr(a5_control, "head_checks", lambda *_: checks)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")
    needs: dict[str, Any] = {name: {"result": "success"} for name in ("changes", "toolchain", a5_control.JOB)}
    needs["changes"]["outputs"] = {"check_id": "41"}
    assert a5_control.attempt_check(REPO, "41", "1234", "3") == str(pending["id"])
    a5_control.finish(REPO, needs)
    assert all(check["status"] == "completed" for check in checks)
    if unfinished == "both":
        assert original["conclusion"] == "cancelled"
    assert pending["conclusion"] == "success" and "/attempts/3)" in pending["output"]["summary"]
    assert a5_control.attempt_check(REPO, "41", "1234", "3") == str(pending["id"])


def test_a5_distributed_requires_its_own_comment(control):
    job = "onboard-tests-a5-distributed"
    full = yaml.safe_load((ROOT / ".github/workflows/ci-full.yml").read_text())["jobs"]
    assert job not in full
    assert job not in full["full-result"]["needs"]
    assert job not in control.REQUIRED
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci-a5-distributed.yml").read_text())
    jobs = workflow["jobs"]
    assert set(jobs) == {"changes", "toolchain", "onboard-tests-a5-distributed", "result"}
    assert set(workflow[True]) == {"issue_comment"}
    assert "github.event.issue.number" in workflow["run-name"]
    assert "contains(" in jobs["changes"]["if"]
    assert jobs["changes"]["permissions"]["actions"] == "read"
    assert jobs["changes"]["concurrency"] == {
        "group": "ci-a5-admission-${{ github.event.issue.number }}",
        "queue": "max",
        "cancel-in-progress": False,
    }
    assert "concurrency" not in workflow
    assert jobs["result"]["concurrency"] == jobs["changes"]["concurrency"]
    observer = yaml.safe_load((ROOT / ".github/workflows/ci-a5-rerun.yml").read_text())["jobs"]
    assert observer["progress"]["needs"] == "route"
    assert observer["progress"]["concurrency"]["group"] == "ci-a5-admission-${{ needs.route.outputs.number }}"
    assert jobs[job]["concurrency"]["cancel-in-progress"] is False
    assert workflow["permissions"] == {}
    for name in ("toolchain", "onboard-tests-a5-distributed"):
        assert jobs[name]["permissions"] == {"contents": "read"}
        assert "checkout_sha != ''" in jobs[name]["if"]
    assert jobs["onboard-tests-a5-distributed"]["needs"] == ["changes", "toolchain"]
    assert jobs["toolchain"]["steps"] == full["toolchain"]["steps"]
    for name in ("changes", "result"):
        checkout = jobs[name]["steps"][0]["with"]
        assert checkout["ref"] == "${{ github.workflow_sha }}"
        assert checkout["persist-credentials"] is False
    assert jobs["result"]["if"].startswith("always()")
    assert set(jobs["result"]["needs"]) == {"changes", "toolchain", "onboard-tests-a5-distributed"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", f"--confcutdir={Path(__file__).parent}"]))
