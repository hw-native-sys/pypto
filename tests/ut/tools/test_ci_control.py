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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
REPO = "owner/repo"
HEAD, BASE, MERGE = "a" * 40, "b" * 40, "c" * 40


@pytest.fixture
def control(monkeypatch):
    spec = importlib.util.spec_from_file_location("ci_control", ROOT / ".github/scripts/ci_control.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("CI_ROUTED_EVENT", raising=False)
    monkeypatch.setenv("CI_ROUTED_HEAD", HEAD)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected subprocess"))
    return module


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
def needs(control):
    result: dict[str, Any] = {name: {"result": "success"} for name in control.REQUIRED | control.OPTIONAL}
    result["changes"]["outputs"] = {"docs_changed": "false", "ptoas_version_changed": "false"}
    result.update({name: {"result": "skipped"} for name in control.CONDITIONAL})
    return result


def test_core_tests_cannot_be_skipped(control, needs):
    needs["system-tests"]["result"] = "skipped"
    with pytest.raises(ValueError, match="system-tests"):
        control.aggregate(needs)


@pytest.mark.parametrize("job", ["prebuild-wheels", "prebuild-wheels-github"])
def test_warmup_failure_does_not_hide_successful_tests(control, needs, job):
    needs[job]["result"] = "failure"
    control.aggregate(needs)


@pytest.mark.parametrize(
    "job,flag", [("docs-examples", "docs_changed"), ("serving-guard", "ptoas_version_changed")]
)
def test_applicable_conditional_job_must_run(control, needs, job, flag):
    needs["changes"]["outputs"][flag] = "true"
    with pytest.raises(ValueError, match=job):
        control.aggregate(needs)
    needs[job]["result"] = "success"
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


def test_changed_pr_rejected_before_checkout(control, pr):
    pr["head"]["sha"] = "d" * 40
    with pytest.raises(ValueError, match="PR changed"):
        control.verify_merge(REPO, pr, HEAD, BASE, MERGE)


@pytest.mark.parametrize("failure", ["", "pending", "dispatch", "bind", "ack"])
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
            writes.append((path, payload))
            if path.endswith("/dispatches"):
                return None if failure == "ack" else {"workflow_run_id": 51}
        elif "/actions/workflows/ci.yml/runs?" in path:
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
        assert len(writes) == {"pending": 0, "dispatch": 1, "bind": 2, "ack": 2}[failure]
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
                "description": "Full CI requested; waiting for dispatch",
                "target_url": f"https://github.com/{REPO}/pull/123",
            },
        ),
        (
            f"repos/{REPO}/actions/workflows/ci.yml/dispatches",
            {
                "ref": "main",
                "return_run_details": True,
                "inputs": {"pr_number": "123", "head_sha": HEAD, "base_sha": BASE, "merge_sha": MERGE},
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
        elif path.endswith("/workflows/ci.yml"):
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
def test_unavailable_merge_never_dispatches(control, monkeypatch, event, pr, conflicting):
    pr["mergeable"] = conflicting

    def request(path):
        assert path.endswith("/pulls/123")
        return pr

    monkeypatch.setattr(control, "api", request)
    with pytest.raises(ValueError, match="conflicting or still being computed"):
        control.dispatch(REPO, event)


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


def test_obsolete_base_cannot_publish_success(control, publication, pr):
    pr["base"]["sha"] = "d" * 40
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
                    "event": "workflow_dispatch",
                    "status": status,
                }
            ]
        }

    monkeypatch.setattr(control, "api", request)
    control.dispatch(REPO, event)
    assert len(queries) == 1
    assert len(writes) == (0 if branch == "main" and status != "completed" else 3)
    if writes:
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
                    "head_branch": "main",
                }
            ]
        }

    monkeypatch.setattr(control, "api", request)
    assert control.newer_request_exists(
        REPO, 123, HEAD, {"created_at": "2026-11-05T01:00:00Z", "run_number": 8}
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
                REPO, 123, HEAD, {"created_at": "2026-11-05T01:00:00Z", "run_number": 8}
            )
        else:
            control.pages("repos/owner/repo/actions/workflows/ci.yml/runs?status=queued", "workflow_runs")
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


@pytest.mark.parametrize("lookup_path", ["/pulls/123", "/workflows/ci.yml"])
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


@pytest.mark.parametrize("lookup_path", ["/pulls/123", "/workflows/ci.yml"])
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
    assert "dispatch failed:" in text
    if cli_error:
        assert "Forbidden &lt;detail&gt; [REDACTED]" in text
        assert "secret-token" not in text
    else:
        assert "PR changed" in text


def test_pr_jobs_cannot_write_github_caches():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
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
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
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
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
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


@pytest.mark.parametrize("lookup_fails", [False, True])
def test_unbound_request_marker_survives_old_publisher(
    control, monkeypatch, publication, publication_event, lookup_fails
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
                "description": "Full CI requested; waiting for dispatch",
                "target_url": f"https://github.com/{REPO}/pull/123",
            }
        ],
    )
    # No replacement run exists yet (including a failed/lost dispatch).
    control.publish(REPO, publication_event)
    assert not publication[3]


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
        "path": control.WORKFLOW,
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
        ("display_title", f"CI lint PR #123 @{HEAD} base:{'d' * 40}"),
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
        pr["base"]["sha"] = "d" * 40
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
            assert path == f"repos/{REPO}/actions/workflows/ci.yml/dispatches"
            return {"workflow_run_id": 51}
        if path == f"repos/{REPO}/actions/runs/51":
            run_reads.append(path)
            return run
        if path == f"repos/{REPO}/actions/workflows/ci.yml":
            return {"id": 99}
        if path == f"repos/{REPO}/pulls/123":
            return pr
        if path == f"repos/{REPO}/git/commits/{MERGE}":
            return {"parents": [{"sha": BASE}, {"sha": HEAD}]}
        assert "/actions/workflows/ci.yml/runs?" in path
        history_reads.append(path)
        return {"workflow_runs": []}  # Both comments see an unindexed run list.

    def pages(path, key):
        assert path == f"repos/{REPO}/commits/{HEAD}/status" and key == "statuses"
        return statuses

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setattr(control, "pages", pages)
    control.dispatch(REPO, event)
    assert len(writes) == 3
    assert statuses[0]["target_url"].endswith("/runs/51/attempts/1")
    control.dispatch(REPO, event)
    assert len(writes) == 3, "The second comment must neither dispatch nor overwrite the binding"
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
        ("display_title", f"CI full PR #123 @{HEAD} base:{'d' * 40}"),
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
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
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
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
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
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", f"--confcutdir={Path(__file__).parent}"]))
