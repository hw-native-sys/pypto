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
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
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
        "path": ".github/workflows/ci-full.yml",
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
            return {
                **payload,
                "id": 200 + len(writes),
                "creator": {"login": "github-actions[bot]"},
            }
        if path.endswith("/runs/50"):
            return copy.deepcopy(run)
        if "/runs/50/attempts/" in path:
            return {**copy.deepcopy(run), "run_attempt": event["workflow_run"]["run_attempt"]}
        if path.endswith("/workflows/ci-full.yml"):
            return {"id": 99, "path": ".github/workflows/ci-full.yml"}
        if path.endswith("/pulls/123"):
            return pr
        raise AssertionError(path)

    def pages(path, key):
        if key == "jobs":
            assert path == f"repos/{REPO}/actions/runs/50/attempts/1/jobs"
            return jobs
        assert key == "statuses" and path == f"repos/{REPO}/commits/{HEAD}/status"
        return checks

    monkeypatch.setattr(progress.control, "api", api)
    monkeypatch.setattr(progress.control, "pages", pages)
    monkeypatch.setattr(progress.control, "newer_status_exists", lambda *args: False)
    monkeypatch.setattr(progress, "poll_budget", lambda *args: (60, 50))
    monkeypatch.setattr(progress.time, "sleep", lambda *args: pytest.fail("Unexpected polling wait"))
    return run, pr, jobs, writes, checks


def test_progress_follows_jobs_without_reposting_unchanged_statuses(progress, monkeypatch, event, harness):
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
    assert [payload["state"] for _, payload, _ in writes] == ["pending", "pending", "success"]
    assert [payload["description"] for _, payload, _ in writes] == [
        "Full CI 50 attempt 1: Queued",
        "Full CI 50 attempt 1: Running",
        "Full CI 50 attempt 1: Success",
    ]
    assert {path for path, _, _ in writes} == {f"repos/{REPO}/statuses/{HEAD}"}
    assert all(not kwargs for _, _, kwargs in writes)
    assert writes[0][1]["context"] == "full / system-tests"
    assert writes[-1][1]["target_url"] == f"https://github.com/{REPO}/actions/runs/50/job/101"


@pytest.mark.parametrize(
    "conclusion,state,label",
    [
        ("failure", "failure", "Failure"),
        ("cancelled", "failure", "Cancelled"),
        ("skipped", "success", "Skipped"),
        ("neutral", "success", "Neutral"),
        ("timed_out", "failure", "Timed out"),
        ("success", "success", "Success"),
        (None, "failure", "Unknown result"),
        ("startup_failure", "failure", "Startup failure"),
    ],
)
def test_completion_describes_actual_job_result(progress, event, harness, conclusion, state, label):
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
    assert writes[0][1]["state"] == state
    assert writes[0][1]["description"].endswith(f": {label}")
    assert writes[1][1]["context"] == "full / unit-tests (ubuntu-latest, 3.10)"


def test_completed_event_reconciles_existing_status_context(progress, event, harness):
    run, _, jobs, writes, checks = harness
    run["status"] = "completed"
    jobs[0].update(status="completed", conclusion="success")
    checks.append(
        {
            "id": 999,
            "context": "full / system-tests",
            "creator": {"login": "github-actions[bot]"},
            "state": "pending",
            "description": "Full CI 50 attempt 1: Running",
        }
    )
    progress.monitor(REPO, event)
    assert writes[0][0] == f"repos/{REPO}/statuses/{HEAD}"
    assert writes[0][1]["context"] == checks[0]["context"]
    assert writes[0][1]["state"] == "success"


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
    assert "checks" not in job["permissions"]
    assert job["permissions"]["statuses"] == "write"
    assert "display_title" not in job["if"]
    assert job["concurrency"]["cancel-in-progress"] is False
    assert job["concurrency"]["queue"] == "max"
    assert job["concurrency"]["group"] == "ci-full-progress-${{ needs.route.outputs.key }}"
    assert job["needs"] == "route"
    assert job["if"] == "needs.route.outputs.key != ''"
    route = workflow["jobs"]["route"]
    assert "concurrency" not in route
    assert route["permissions"] == {"contents": "read", "actions": "read"}
    assert route["steps"][-1]["run"].endswith("ci_progress.py route")
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
    existing = [
        {
            "id": 900 + index,
            "context": f"full / {name}",
            "description": "Full CI 50 attempt 1: Failure",
            "creator": {"login": "github-actions[bot]"},
        }
        for index, name in enumerate(["unit-tests", "system-tests"])
    ]

    def pages(path, key):
        if key == "statuses":
            return existing
        assert key == "jobs"
        if path.endswith("/attempts/1/jobs"):
            return old_jobs
        assert path.endswith("/attempts/2/jobs")
        return new_jobs

    monkeypatch.setattr(progress.control, "pages", pages)
    progress.monitor(REPO, event)
    assert {payload["context"] for _, payload, _ in writes} == {"full / unit-tests", "full / system-tests"}
    assert all(payload["state"] == "success" for _, payload, _ in writes)
    assert all(payload["description"] == "Full CI 50 attempt 2: Success" for _, payload, _ in writes)
    assert {path for path, _, _ in writes} == {f"repos/{REPO}/statuses/{HEAD}"}
    assert {payload["target_url"].rsplit("/", 1)[1] for _, payload, _ in writes} == {"100", "102"}


def test_repeated_reruns_keep_one_context_through_running_and_completed_states(
    progress, monkeypatch, event, harness
):
    run, _, jobs, writes, checks = harness
    original_api = progress.control.api

    def api(path, payload=None, **kwargs):
        result = original_api(path, payload, **kwargs)
        if payload is not None:
            checks.append(result)
        return result

    monkeypatch.setattr(progress.control, "api", api)
    monkeypatch.setattr(progress.control, "effective_jobs", lambda *args: jobs)
    run["status"] = "completed"
    jobs[0].update(status="completed", conclusion="failure")
    progress.monitor(REPO, event)
    # Status writes append history, while one stable context identifies the
    # single visible row. A native rerun must not introduce an attempt suffix.
    for attempt in (2, 3):
        event["workflow_run"]["run_attempt"] = attempt
        run.update(run_attempt=attempt, status="in_progress")
        jobs[0].update(id=101 + attempt, status="in_progress", conclusion=None)

        def complete(seconds):
            latest = progress.latest_statuses(checks)["full / system-tests"]
            assert latest["state"] == "pending"
            assert latest["description"].endswith(": Running")
            run["status"] = "completed"
            jobs[0].update(status="completed", conclusion="success")

        monkeypatch.setattr(progress.time, "sleep", complete)
        progress.monitor(REPO, event)
        visible = progress.latest_statuses(checks)
        assert len(visible) == 1
        latest = visible["full / system-tests"]
        assert latest["description"] == f"Full CI 50 attempt {attempt}: Success"
        assert latest["state"] == "success"
        assert latest["target_url"].endswith(f"/job/{101 + attempt}")
    assert len(writes) == 5
    assert all(not kwargs for _, _, kwargs in writes)


@pytest.mark.parametrize("reverse", [False, True])
def test_latest_status_is_independent_of_page_order(progress, reverse):
    records = [
        {
            "id": check_id,
            "context": "full / matrix (a:b)",
            "description": f"Full CI 50 attempt {attempt}: Success",
            "creator": {"login": "github-actions[bot]"},
        }
        for check_id, attempt in [(999, 1), (800, 2), (801, 2)]
    ]
    if reverse:
        records.reverse()
    selected = progress.latest_statuses(records)
    assert selected["full / matrix (a:b)"]["id"] == 999


@pytest.mark.parametrize(
    "context,creator",
    [
        ("ci/full", "github-actions[bot]"),
        ("full / system-tests", "another-app[bot]"),
        ("unrelated", "github-actions[bot]"),
    ],
)
def test_status_cache_ignores_other_contexts_and_publishers(progress, context, creator):
    assert (
        progress.latest_statuses([{"id": 900, "context": context, "creator": {"login": creator}}])
        == {}
    )


def test_future_status_attempt_prevents_stale_projection(progress, event, harness):
    run, _, _, writes, checks = harness
    run["status"] = "completed"
    checks.append(
        {
            "id": 900,
            "context": "full / system-tests",
            "description": "Full CI 50 attempt 2: Success",
            "creator": {"login": "github-actions[bot]"},
        }
    )
    progress.monitor(REPO, event)
    assert not writes


def test_lost_create_response_is_read_back_before_retry(progress, monkeypatch, event, harness):
    run, _, jobs, writes, checks = harness
    run["status"] = "completed"
    jobs[0].update(status="completed", conclusion="success")
    original_api = progress.control.api
    failed = False

    def api(path, payload=None, **kwargs):
        nonlocal failed
        result = original_api(path, payload, **kwargs)
        if payload is not None and not failed:
            failed = True
            checks.append(result)
            raise subprocess.CalledProcessError(1, ["gh", "api"])
        return result

    waits = []
    monkeypatch.setattr(progress.control, "api", api)
    monkeypatch.setattr(progress.time, "sleep", waits.append)
    progress.monitor(REPO, event)
    assert waits == [60]
    assert len(writes) == 1


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


@pytest.mark.parametrize("manual", [False, True])
def test_monitor_backs_off_on_transient_api_failure(progress, monkeypatch, event, harness, manual):
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
    if manual:
        # The first REST read resolves the run ID, before any monitoring poll.
        monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
        monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
        event = {"inputs": {"run_id": "50"}}
    progress.monitor(REPO, event)
    assert waits == [60]
    assert harness[3]


def test_manual_resolution_stops_after_bounded_retries(progress, monkeypatch, tmp_path):
    reads = []
    waits = []

    def unavailable(path):
        reads.append(path)
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    monkeypatch.setattr(progress.control, "api", unavailable)
    monkeypatch.setattr(progress.time, "sleep", waits.append)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    with pytest.raises(subprocess.CalledProcessError):
        progress.monitor(REPO, {"inputs": {"run_id": "50"}})
    assert len(reads) == 5
    assert waits == [60, 120, 240, 300]
    assert not output.exists(), "An unresolved run must not trigger status publication"


def test_non_full_main_ci_does_not_request_status_publication(
    progress, monkeypatch, event, harness, tmp_path
):
    run, _, _, writes, _ = harness
    run["display_title"] = "CI workflow_dispatch main"
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    progress.monitor(REPO, event)
    assert not output.exists()
    assert not writes


@pytest.mark.parametrize("manual", [False, True])
def test_identified_full_source_keeps_fallback_after_monitor_failure(
    progress, monkeypatch, event, harness, tmp_path, manual
):
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    if manual:
        monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
        monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
        event = {"inputs": {"run_id": "50"}}

    def unavailable(*args):
        raise subprocess.CalledProcessError(1, ["gh", "api"])

    waits = []
    monkeypatch.setattr(progress, "poll_budget", unavailable)
    monkeypatch.setattr(progress.time, "sleep", waits.append)
    with pytest.raises(subprocess.CalledProcessError):
        progress.monitor(REPO, event)
    assert output.read_text() == "source_run=50\n"
    assert waits == [60, 120, 240, 300]
    assert not harness[3]


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
    assert writes[0][0] == f"repos/{REPO}/statuses/{HEAD}"
    assert writes[0][1]["state"] == "success"


def test_explicit_dispatch_mirrors_existing_results(progress, monkeypatch, harness):
    run, _, jobs, writes, _ = harness
    run["status"] = "completed"
    jobs[0].update(status="completed", conclusion="success")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    progress.monitor(REPO, {"inputs": {"run_id": "50"}})
    assert len(writes) == 1
    path, payload, _ = writes[0]
    assert path == f"repos/{REPO}/statuses/{HEAD}"
    assert payload["state"] == "success"


@pytest.mark.parametrize("ref", ["refs/heads/feature", "refs/tags/main", ""])
def test_explicit_progress_dispatch_rejects_untrusted_ref(progress, monkeypatch, ref):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", ref)
    with pytest.raises(progress.control.UntrustedRunError, match="must run from main"):
        progress.monitor(REPO, {"inputs": {"run_id": "50"}})


def test_progress_reconciles_legacy_full_run(progress, monkeypatch, event, harness):
    run, _, jobs, writes, _ = harness
    for source in (event["workflow_run"], run):
        source["path"] = progress.control.LINT_WORKFLOW
    event["workflow"]["path"] = progress.control.LINT_WORKFLOW
    original = progress.control.api

    def api(path, payload=None, **kwargs):
        if path.endswith("/workflows/ci.yml"):
            return {"id": 99, "path": progress.control.LINT_WORKFLOW}
        return original(path, payload, **kwargs)

    monkeypatch.setattr(progress.control, "api", api)
    run["status"] = "completed"
    jobs[0].update(status="completed", conclusion="success")
    progress.monitor(REPO, event)
    assert writes[-1][1]["state"] == "success"


def test_long_matrix_names_do_not_collide(progress):
    prefix = "unit-tests (" + "a" * 120
    first = progress.job_context(prefix + ", python3.10)")
    second = progress.job_context(prefix + ", python3.11)")
    assert len(first) == len(second) == 100
    assert first != second
    assert progress.job_context("unit-tests (ubuntu, 3.10)") == "full / unit-tests (ubuntu, 3.10)"


def test_new_full_run_reuses_context_and_replaces_log_link(progress, monkeypatch):
    jobs = [{"id": 101, "name": "system-tests", "status": "completed", "conclusion": "success"}]
    monkeypatch.setattr(progress.control, "effective_jobs", lambda *args: jobs)
    writes = []

    def api(path, payload):
        writes.append((path, payload))
        return {**payload, "id": len(writes)}

    monkeypatch.setattr(progress.control, "api", api)
    statuses = {}
    for run_id in (50, 51):
        progress.sync_jobs(REPO, HEAD, {"id": run_id, "run_attempt": 1}, statuses)
    assert len(statuses) == 1
    assert {p["context"] for _, p in writes} == {"full / system-tests"}
    assert statuses["full / system-tests"]["target_url"].endswith("/runs/51/job/101")


def test_publisher_detects_server_rewriting_log_link(progress, monkeypatch, event, harness):
    run, _, jobs, _, _ = harness
    run["status"] = "completed"
    jobs[0].update(status="completed", conclusion="success")
    original = progress.control.api

    def api(path, payload=None, **kwargs):
        result = original(path, payload, **kwargs)
        if payload is not None:
            result["target_url"] = "https://github.com/incorrect-workflow"
        return result

    monkeypatch.setattr(progress.control, "api", api)
    with pytest.raises(ValueError, match="did not preserve"):
        progress.monitor(REPO, event)


@pytest.mark.parametrize("run_id,attempt", [(50, 1), (51, 1), (50, 2)])
def test_route_groups_every_run_and_attempt_for_one_pr_commit(
    progress, monkeypatch, harness, tmp_path, run_id, attempt
):
    run = harness[0]
    run.update(id=run_id, run_attempt=attempt)
    original = progress.control.api

    def api(path):
        if path == f"repos/{REPO}/actions/runs/{run_id}":
            return copy.deepcopy(run)
        return original(path)

    monkeypatch.setattr(progress.control, "api", api)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    progress.route(REPO, {"inputs": {"run_id": str(run_id)}})
    outputs = dict(line.split("=", 1) for line in target.read_text().splitlines())
    assert outputs["key"] == f"123-{HEAD}"
    assert not harness[3], "Routing must not write any status"


@pytest.mark.parametrize("recovers", [True, False])
def test_route_retries_api_failure_before_entering_writer_queue(
    progress, monkeypatch, harness, tmp_path, recovers
):
    original = progress.control.api
    reads = []
    waits = []

    def api(path):
        if path.endswith("/runs/50"):
            reads.append(path)
            if not recovers or len(reads) == 1:
                raise subprocess.CalledProcessError(1, ["gh", "api"])
        return original(path)

    monkeypatch.setattr(progress.control, "api", api)
    monkeypatch.setattr(progress.time, "sleep", waits.append)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    if recovers:
        progress.route(REPO, {"inputs": {"run_id": "50"}})
        assert f"key=123-{HEAD}\n" in target.read_text()
        assert waits == [60]
    else:
        with pytest.raises(subprocess.CalledProcessError):
            progress.route(REPO, {"inputs": {"run_id": "50"}})
        assert not target.exists()
        assert len(reads) == 5
        assert waits == [60, 120, 240, 300]
    assert not harness[3]


def test_route_does_not_queue_untrusted_dispatch(progress, monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/feature")
    target = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    with pytest.raises(progress.control.UntrustedRunError):
        progress.route(REPO, {"inputs": {"run_id": "50"}})
    assert not target.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", f"--confcutdir={Path(__file__).parent}"]))
