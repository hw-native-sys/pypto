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
import subprocess
from pathlib import Path
from typing import Any

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
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected subprocess"))
    return module


@pytest.fixture
def pr():
    return {
        "state": "open",
        "number": 123,
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
        "comment": {"body": "/ci full", "user": {"login": "maintainer"}},
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
    "kind,enabled,expected",
    [
        ("pull_request", "true", "false"),
        ("pull_request", "", "true"),
        ("push", "true", "true"),
        ("workflow_dispatch", "true", "true"),
    ],
)
def test_rollout_preserves_legacy_and_main_runs(control, monkeypatch, tmp_path, kind, enabled, expected):
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    monkeypatch.setenv("GITHUB_EVENT_NAME", kind)
    monkeypatch.setenv("GITHUB_SHA", MERGE)
    monkeypatch.setenv("CI_COMMENT_TIERS_ENABLED", enabled)
    control.prepare(REPO, {})
    assert target.read_text() == f"full={expected}\ncheckout_sha={MERGE}\n"


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


def test_dispatch_authorizes_and_pins_fork_pr(control, monkeypatch, event, pr):
    writes = []

    def api(path, payload=None):
        if payload is not None:
            writes.append((path, payload))
        elif path.endswith("/permission"):
            return {"permission": "write"}
        elif "/pulls/" in path:
            return pr
        elif "/git/commits/" in path:
            return {"parents": [{"sha": BASE}, {"sha": HEAD}]}
        else:
            raise AssertionError(path)

    monkeypatch.setattr(control, "api", api)
    monkeypatch.setattr(control, "pages", lambda *a: [])
    # Empty pull_request data is still a PR marker; the issue API link normally
    # has fields, but identity should depend on its presence rather than contents.
    event["issue"]["pull_request"] = {"url": "unused"}
    control.dispatch(REPO, event)
    assert writes == [
        (
            f"repos/{REPO}/actions/workflows/ci.yml/dispatches",
            {
                "ref": "main",
                "inputs": {"pr_number": "123", "head_sha": HEAD, "base_sha": BASE, "merge_sha": MERGE},
            },
        )
    ]


def test_contributor_association_does_not_grant_full_access(control, monkeypatch, event):
    event["issue"]["pull_request"] = {"url": "unused"}
    event["comment"]["author_association"] = "CONTRIBUTOR"
    monkeypatch.setattr(control, "api", lambda path: {"permission": "read"})
    with pytest.raises(ValueError, match="Only collaborators"):
        control.dispatch(REPO, event)


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
            return {"id": 99}
        elif path.endswith("/pulls/123"):
            return pr
        elif "event=workflow_dispatch" in path:
            assert "created=>=2026-11-05T01:00:00Z" in path
            return {"workflow_runs": candidates}
        else:
            raise AssertionError(path)

    def pages(path, key):
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


def test_new_head_cannot_receive_old_success(control, publication, pr):
    pr["head"]["sha"] = "d" * 40
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not publication[3]


def test_old_request_cannot_overwrite_newer_request(control, publication):
    run, _, candidates, writes = publication
    newer = copy.deepcopy(run)
    newer.update(id=51, run_number=9, status="queued")
    candidates.append(newer)
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not writes


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
        if name not in {"changes", "toolchain", "pre-commit", "clang-tidy", "full-result"}:
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
                        "github.event_name == 'pull_request' && vars.CI_COMMENT_TIERS_ENABLED == 'true' "
                        "&& github.event.pull_request.base.sha || github.workflow_sha"
                    )
                assert options["ref"] == "${{ " + expected + " }}"
                assert options["persist-credentials"] is False
            if step.get("uses") == "./.github/actions/setup-ci-job":
                assert step["with"]["checkout-ref"] == "${{ needs.changes.outputs.checkout_sha }}"
    # The full aggregate itself is not the required context: it is attached to
    # main in dispatch runs. Only the isolated publisher writes the PR status.
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
        if path.endswith("/permission"):
            return {"permission": "write"}
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
    control.prepare(
        REPO, {"inputs": {"pr_number": "123", "head_sha": HEAD, "base_sha": BASE, "merge_sha": MERGE}}
    )
    assert target.read_text() == f"full=true\ncheckout_sha={MERGE}\n"


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


def test_closed_pr_result_is_ignored(control, publication, pr):
    pr["state"] = "closed"
    control.publish(REPO, {"workflow_run": {"id": 50}})
    assert not publication[3]


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


@pytest.mark.parametrize("status", ["queued", "in_progress", "requested", "waiting", "pending"])
def test_every_unfinished_state_blocks_duplicate_dispatch(control, monkeypatch, event, pr, status):
    queries = []

    def request(path, payload=None):
        assert payload is None, "An unfinished request must prevent dispatch"
        if path.endswith("/permission"):
            return {"permission": "write"}
        if path.endswith("/pulls/123"):
            return pr
        assert path.endswith(f"/git/commits/{MERGE}")
        return {"parents": [{"sha": BASE}, {"sha": HEAD}]}

    def pages(path, key):
        queries.append(path)
        assert key == "workflow_runs"
        assert "&status=" in path, "Never scan the entire historical run list"
        return [{"display_title": f"CI full PR #123 @{HEAD} base:{BASE}"}] if path.endswith(status) else []

    monkeypatch.setattr(control, "api", request)
    monkeypatch.setattr(control, "pages", pages)
    control.dispatch(REPO, event)
    assert any(path.endswith(f"status={status}") for path in queries)


def test_publication_history_is_bounded_by_creation_time(control, monkeypatch):
    calls = []

    def request(path):
        calls.append(path)
        assert "&created=>=2026-11-05T01:00:00Z&per_page=100" in path
        if path.endswith("page=1"):
            # Unrelated runs can fill pages. Do not mistake them for requests
            # for this PR, or assume a short history that fits in one page.
            return {"workflow_runs": [{"display_title": "Manual main", "run_number": 90}] * 100}
        assert path.endswith("page=2")
        return {
            "workflow_runs": [{"display_title": f"CI full PR #123 @{HEAD} base:{BASE}", "run_number": 10}]
        }

    monkeypatch.setattr(control, "api", request)
    assert control.newer_request_exists(
        REPO, 123, HEAD, {"created_at": "2026-11-05T01:00:00Z", "run_number": 8}
    )
    assert len(calls) == 2


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
        raise ValueError("Only collaborators may request full CI")

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
        assert "Only collaborators" in text


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
    assert "key={match[1]}-{match[2]}" in publisher["jobs"]["route"]["steps"][0]["run"]


def test_tiered_pr_controller_uses_trusted_base_without_changing_trigger():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    # PyYAML's YAML 1.1 loader parses the unquoted Actions key "on" as True.
    assert "pull_request" in workflow[True]
    controller = workflow["jobs"]["changes"]["steps"][0]["with"]
    assert controller["ref"] == (
        "${{ github.event_name == 'pull_request' && vars.CI_COMMENT_TIERS_ENABLED == 'true' "
        "&& github.event.pull_request.base.sha || github.workflow_sha }}"
    )
    assert controller["sparse-checkout"] == ".github/scripts"
    assert controller["persist-credentials"] is False
