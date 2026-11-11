# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test PR job visibility without runners, GitHub writes, or polling delays."""

import copy
import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
REPO = "owner/repo"
HEAD, BASE = "a" * 40, "b" * 40


@pytest.fixture
def progress(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / ".github/scripts"))
    spec = importlib.util.spec_from_file_location("ci_progress", ROOT / ".github/scripts/ci_progress.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.control, "api", lambda *a, **k: pytest.fail("Unexpected GitHub API call"))
    return module


@pytest.fixture
def event():
    repository = {"id": 1, "full_name": REPO}
    run = {
        "id": 50,
        "run_number": 8,
        "run_attempt": 1,
        "workflow_id": 99,
        "display_title": f"CI full PR #123 @{HEAD} base:{BASE}",
        "event": "workflow_dispatch",
        "path": ".github/workflows/ci.yml",
        "head_branch": "main",
        "repository": repository,
        "head_repository": repository,
        "status": "in_progress",
        "conclusion": None,
    }
    return {
        "action": "in_progress",
        "workflow_run": run,
        "repository": repository,
        "workflow": {"id": 99, "path": run["path"]},
    }


@pytest.fixture
def harness(progress, monkeypatch, event):
    run = copy.deepcopy(event["workflow_run"])
    pr = {
        "state": "open",
        "head": {"sha": HEAD, "repo": {"id": 2}},
        "base": {"sha": BASE, "ref": "main", "repo": {"full_name": REPO}},
    }
    jobs = [{"id": 101, "name": "system-tests", "status": "queued", "conclusion": None}]
    writes = []
    checks = []

    def api(path, payload=None, **kwargs):
        if payload is not None:
            writes.append((path, payload.copy(), kwargs))
            return {"id": 200 + len(writes)}
        if path.endswith("/runs/50"):
            return copy.deepcopy(run)
        if "/runs/50/attempts/" in path:
            return {**copy.deepcopy(run), "run_attempt": event["workflow_run"]["run_attempt"]}
        if path.endswith("/workflows/ci.yml"):
            return {"id": 99}
        if path.endswith("/pulls/123"):
            return pr
        raise AssertionError(path)

    def pages(path, key):
        if key == "jobs":
            assert path == f"repos/{REPO}/actions/runs/50/attempts/1/jobs"
            return jobs
        assert key == "check_runs" and f"/commits/{HEAD}/check-runs?filter=all" in path
        return checks

    monkeypatch.setattr(progress.control, "api", api)
    monkeypatch.setattr(progress.control, "pages", pages)
    monkeypatch.setattr(progress.control, "newer_status_exists", lambda *args: False)
    monkeypatch.setattr(progress, "poll_budget", lambda *args: (60, 50))
    monkeypatch.setattr(progress.time, "sleep", lambda *args: pytest.fail("Unexpected polling wait"))
    return run, pr, jobs, writes, checks


def test_progress_follows_jobs_and_does_not_duplicate_unchanged_checks(progress, monkeypatch, event, harness):
    run, _, jobs, writes, _ = harness
    transitions = iter(["in_progress", "in_progress", "completed"])

    def advance(seconds):
        assert seconds == 60
        jobs[0]["status"] = next(transitions)
        if jobs[0]["status"] == "completed":
            jobs[0]["conclusion"] = "success"
            run["status"] = "completed"

    monkeypatch.setattr(progress.time, "sleep", advance)
    progress.monitor(REPO, event)
    assert [payload["status"] for _, payload, _ in writes] == ["queued", "in_progress", "completed"]
    assert writes[0][1]["head_sha"] == HEAD
    assert writes[0][1]["name"] == "full / system-tests"
    assert writes[1][2] == {"method": "PATCH"}
    assert writes[-1][1]["conclusion"] == "success"
    assert writes[-1][1]["details_url"].endswith("/runs/50/job/101")


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "skipped", "timed_out", "success"])
def test_completion_preserves_individual_job_results(progress, event, harness, conclusion):
    run, _, jobs, writes, _ = harness
    run["status"] = "completed"
    jobs[0].update(status="completed", conclusion=conclusion)
    jobs.append(
        {
            "id": 102,
            "name": "unit-tests (ubuntu-latest, 3.10)",
            "status": "completed",
            "conclusion": "success",
        }
    )
    progress.monitor(REPO, event)
    assert writes[0][1]["conclusion"] == conclusion
    assert writes[1][1]["name"] == "full / unit-tests (ubuntu-latest, 3.10)"


def test_completed_event_reconciles_existing_checks(progress, event, harness):
    run, _, jobs, writes, checks = harness
    run["status"] = "completed"
    jobs[0].update(status="completed", conclusion="success")
    checks.append(
        {
            "id": 999,
            "external_id": "ci-full:50:1:system-tests",
            "app": {"slug": "github-actions"},
            "status": "in_progress",
        }
    )
    progress.monitor(REPO, event)
    assert writes[0][0] == f"repos/{REPO}/check-runs/999"
    assert writes[0][2] == {"method": "PATCH"}


@pytest.mark.parametrize("change", ["attempt", "head", "base", "newer"])
def test_obsolete_monitor_cannot_publish_new_revision(progress, monkeypatch, event, harness, change):
    run, pr, _, writes, _ = harness
    if change == "attempt":
        run["run_attempt"] = 2
    elif change in {"head", "base"}:
        pr[change]["sha"] = "d" * 40
    else:
        monkeypatch.setattr(progress.control, "newer_status_exists", lambda *args: True)
    progress.monitor(REPO, event)
    assert not writes


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", ".github/workflows/evil.yml"),
        ("head_branch", "feature"),
        ("event", "pull_request"),
        ("head_repository", {"id": 2}),
    ],
)
def test_untrusted_progress_event_never_writes(progress, event, harness, field, value):
    event["workflow_run"][field] = value
    with pytest.raises(progress.control.UntrustedRunError):
        progress.monitor(REPO, event)
    assert not harness[3]


def test_monitor_window_is_bounded(progress, monkeypatch, event, harness):
    clock = iter([0, progress.MONITOR_SECONDS])
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(clock))
    progress.monitor(REPO, event)
    assert len(harness[3]) == 0


def test_progress_workflow_is_isolated_from_test_code():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci-full-progress.yml").read_text())
    assert workflow[True]["workflow_run"]["types"] == ["in_progress", "completed"]
    job = workflow["jobs"]["progress"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["permissions"]["checks"] == "write"
    assert job["permissions"]["statuses"] == "read"
    assert "display_title" not in job["if"]
    assert job["concurrency"]["cancel-in-progress"] is False
    checkout = job["steps"][0]["with"]
    assert checkout["ref"] == "${{ github.workflow_sha }}"
    assert checkout["sparse-checkout"] == ".github/scripts"
    assert checkout["persist-credentials"] is False


def test_rerun_progress_includes_retained_successful_jobs(progress, monkeypatch, event, harness):
    run, _, _, writes, _ = harness
    run.update(run_attempt=2, status="completed")
    event["workflow_run"].update(run_attempt=2)
    old_jobs = [
        {"id": 100, "name": "unit-tests", "status": "completed", "conclusion": "success"},
        {"id": 101, "name": "system-tests", "status": "completed", "conclusion": "failure"},
    ]
    new_jobs = [{"id": 102, "name": "system-tests", "status": "completed", "conclusion": "success"}]

    def pages(path, key):
        if key == "check_runs":
            return []
        assert key == "jobs"
        if path.endswith("/attempts/1/jobs"):
            return old_jobs
        assert path.endswith("/attempts/2/jobs")
        return new_jobs

    monkeypatch.setattr(progress.control, "pages", pages)
    progress.monitor(REPO, event)
    assert {payload["name"] for _, payload, _ in writes} == {"full / unit-tests", "full / system-tests"}
    assert all(payload["conclusion"] == "success" for _, payload, _ in writes)
    assert all(payload["external_id"].startswith("ci-full:50:2:") for _, payload, _ in writes)


@pytest.mark.parametrize("active", [1, 2, 10])
def test_poll_budget_scales_with_concurrent_monitors(progress, monkeypatch, active):
    monkeypatch.setattr(progress.time, "time", lambda: 0)

    def api(path):
        if path == "rate_limit":
            return {"resources": {"core": {"remaining": 1000, "reset": 3600}}}
        assert "ci-full-progress.yml/runs?status=in_progress" in path
        return {"total_count": active}

    monkeypatch.setattr(progress.control, "api", api)
    interval, writes = progress.poll_budget(REPO, 1)
    assert active * 7 * 3600 / interval <= progress.POLL_REQUESTS_PER_HOUR
    assert 0 < writes <= 50


def test_poll_budget_reserves_capacity_for_gate_writer(progress, monkeypatch):
    def api(path):
        assert path == "rate_limit", "No workflow requests while quota is reserved"
        return {"resources": {"core": {"remaining": 200, "reset": 3600}}}

    monkeypatch.setattr(progress.control, "api", api)
    monkeypatch.setattr(progress.time, "time", lambda: 0)
    assert progress.poll_budget(REPO, 1) == (3601, 0)


@pytest.mark.parametrize("remaining,active,attempt", [(201, 1, 1), (207, 1, 1), (208, 1, 2), (215, 2, 1)])
def test_poll_budget_wakes_at_reset_when_no_write_fits(progress, monkeypatch, remaining, active, attempt):
    def api(path):
        if path == "rate_limit":
            return {"resources": {"core": {"remaining": remaining, "reset": 3600}}}
        assert "ci-full-progress.yml/runs?status=in_progress" in path
        return {"total_count": active}

    monkeypatch.setattr(progress.control, "api", api)
    monkeypatch.setattr(progress.time, "time", lambda: 0)
    # With 201 remaining requests the proportional delay was seven hours,
    # so a five-hour completion monitor never got to publish the final checks.
    assert progress.poll_budget(REPO, attempt) == (3601, 0)


def test_monitor_backs_off_on_transient_api_failure(progress, monkeypatch, event, harness):
    real_api = progress.control.api
    failed = False
    waits = []
    harness[0]["status"] = "completed"

    def api(path, *args, **kwargs):
        nonlocal failed
        if path.endswith("/runs/50") and not failed:
            failed = True
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return real_api(path, *args, **kwargs)

    monkeypatch.setattr(progress.control, "api", api)
    monkeypatch.setattr(progress.time, "sleep", waits.append)
    progress.monitor(REPO, event)
    assert waits == [60]
    assert harness[3]


def test_completion_reconciles_jobs_across_write_budget(progress, monkeypatch, event, harness):
    run, _, jobs, writes, _ = harness
    run["status"] = "completed"
    jobs[0].update(status="completed", conclusion="success")
    jobs.append({"id": 102, "name": "unit-tests", "status": "completed", "conclusion": "success"})
    monkeypatch.setattr(progress, "poll_budget", lambda *args: (60, 1))
    waits = []
    monkeypatch.setattr(progress.time, "sleep", waits.append)
    progress.monitor(REPO, event)
    assert waits == [60]
    assert len(writes) == 2


@pytest.mark.parametrize("title", [None, "", "CI"])
def test_progress_resolves_static_or_missing_webhook_title(progress, event, harness, title):
    run, _, jobs, writes, _ = harness
    run["status"] = "completed"
    jobs[0].update(status="completed", conclusion="success")
    if title is None:
        event["workflow_run"].pop("display_title")
    else:
        event["workflow_run"]["display_title"] = title
    progress.monitor(REPO, event)
    assert len(writes) == 1
    assert writes[0][1]["head_sha"] == HEAD
    assert writes[0][1]["conclusion"] == "success"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", f"--confcutdir={Path(__file__).parent}"]))
