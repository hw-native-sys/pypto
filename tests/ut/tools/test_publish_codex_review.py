# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression tests for automated PR approval boundaries, without GitHub writes."""

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

HEAD = "a" * 40
BASE = "b" * 40


@pytest.fixture
def publisher():
    """Load the trusted publisher without importing the compiler extension."""
    path = Path(__file__).resolve().parents[3] / ".github/scripts/publish_codex_review.py"
    spec = importlib.util.spec_from_file_location("publish_codex_review", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def review_file(tmp_path):
    """Create a complete clean review artifact for publication tests."""
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"verdict": "pass", "summary": "Reviewed all changes.", "findings": []}))
    return path


@pytest.fixture
def api(publisher, monkeypatch):
    """Model GitHub reads, writes, and revision races without network access."""
    state = {
        "pr": {
            "state": "open",
            "draft": False,
            "user": {"login": "contributor"},
            "head": {"sha": HEAD},
            "base": {"sha": BASE, "ref": "main"},
            "changed_files": 1,
        },
        "files": [{"filename": "src/example.cpp"}],
        "rules": [
            {
                "type": "pull_request",
                "parameters": {
                    "dismiss_stale_reviews_on_push": True,
                    "required_approving_review_count": 1,
                },
            },
            {
                "type": "required_status_checks",
                "parameters": {
                    "strict_required_status_checks_policy": True,
                    "required_status_checks": [{"context": "unit-tests", "integration_id": 15368}],
                },
            },
        ],
        "reviews": [],
        "posted": [],
        "dismissed": [],
        "reads": 0,
    }

    def request(endpoint, payload=None, *, paginate=False):
        """Return API fixtures and inject the requested publication race."""
        if payload is not None:
            state["posted"].append(payload)
            if state.get("retarget_at") == "post":
                state["pr"]["base"]["ref"] = "release"
            return {"id": 42}
        if "/files?" in endpoint:
            assert paginate
            return state["files"]
        if "/rules/" in endpoint:
            assert paginate
            return state["rules"]
        if "/reviews?" in endpoint:
            return state["reviews"]
        state["reads"] += 1
        if state.get("retarget_at") == state["reads"]:
            state["pr"]["base"]["ref"] = "release"
        if state.get("change_at") == state["reads"]:
            state["pr"]["head"]["sha"] = "c" * 40
        return json.loads(json.dumps(state["pr"]))

    def run(command, **kwargs):
        """Record approval dismissals instead of invoking GitHub."""
        assert command[-4:] == ["--method", "PUT", "--input", "-"]
        state["dismissed"].append(command)

    monkeypatch.setattr(publisher, "github_api", request)
    monkeypatch.setattr(publisher.subprocess, "run", run)
    return state


def test_clean_review_approves_exact_commit(publisher, review_file, api):
    """Bind a valid approval to exactly the reviewed head commit."""
    assert publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True).startswith("Approved")
    assert api["posted"][0]["event"] == "APPROVE"
    assert api["posted"][0]["commit_id"] == HEAD


def test_actions_bot_authored_pr_is_comment_only(publisher, review_file, api):
    """Keep a clean review visible when the bot cannot approve its own PR."""
    api["pr"]["user"]["login"] = "github-actions[bot]"
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert api["posted"][0]["event"] == "COMMENT"
    assert "cannot approve its own pull request" in api["posted"][0]["body"]


@pytest.mark.parametrize("body", [None, ""])
def test_bodyless_prior_bot_approval_does_not_block_review(publisher, review_file, api, body):
    """Skip unmarked bodyless approvals while replacing this workflow's review."""
    api["reviews"] = [
        {"id": 1, "user": {"login": "github-actions[bot]"}, "state": "APPROVED", "body": body},
        {"id": 2, "user": {"login": "github-actions[bot]"}, "state": "APPROVED", "body": publisher.MARKER},
    ]
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert len(api["dismissed"]) == 1
    assert "/reviews/2/dismissals" in api["dismissed"][0][2]
    assert api["posted"][0]["event"] == "APPROVE"


@pytest.mark.parametrize(
    "verdict,findings",
    [
        ("findings", [{"title": "[P1] Bug", "body": "src/example.cpp:10 has an overflow."}]),
        ("incomplete", []),
    ],
)
def test_nonpassing_review_never_approves(publisher, review_file, api, verdict, findings):
    """Publish findings and incomplete reviews without approving."""
    review_file.write_text(
        json.dumps({"verdict": verdict, "summary": "Needs attention", "findings": findings})
    )
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert api["posted"][0]["event"] == "COMMENT"


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "Everything looks good",
        "{}",
        "[]",
        "null",
        '{"verdict":"pass"}',
        '{"verdict":"pass","summary":"ok","findings":[],"extra":true}',
        '{"verdict":"pass","summary":"ok","findings":[],"verdict":"pass"}',
        '{"verdict":"pass","summary":"ok","findings":[{"title":"bug","body":"bad"}]}',
        '{"verdict":"findings","summary":"ok","findings":[]}',
        '{"verdict":"pass","summary":"","findings":[]}',
        '{"verdict":"pass","summary":"ok","findings":null}',
        '{"verdict":"approved","summary":"ok","findings":[]}',
        '{"verdict":"findings","summary":"ok","findings":[{"title":"bug"}]}',
        "x" * 60001,
    ],
)
def test_invalid_results_cannot_approve(publisher, review_file, api, contents):
    """Reject malformed or inconsistent artifacts before publication."""
    review_file.write_text(contents)
    with pytest.raises(ValueError):
        publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert not api["posted"]


def test_symlink_rejected(publisher, review_file, api, tmp_path):
    """Reject an artifact symlink before reading model output."""
    link = tmp_path / "link"
    link.symlink_to(review_file)
    with pytest.raises(ValueError, match="regular file"):
        publisher.publish(link, "owner/repo", "12", HEAD, BASE, "main", True)
    assert not api["posted"]


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/new.yml",
        ".github/scripts/new.py",
        "AGENTS.md",
        "src/AGENTS.md",
        "AGENTS.override.md",
        "src/AGENTS.override.md",
        ".codex/config.toml",
        ".claude/rules/review.md",
        ".agents/skills/policy",
        ".gitmodules",
    ],
)
@pytest.mark.parametrize("rename", [False, True])
def test_policy_changes_require_human_review(publisher, review_file, api, path, rename):
    """Withhold approval for policy paths and their renamed originals."""
    api["files"] = [{"filename": "renamed.txt", "previous_filename": path} if rename else {"filename": path}]
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert api["posted"][0]["event"] == "COMMENT"


@pytest.mark.parametrize(
    "gate",
    [
        "disabled",
        "no_rules",
        "stale_allowed",
        "zero_approvals",
        "missing_approval_count",
        "no_check_rule",
        "non_strict_checks",
        "empty_checks",
        "truncated_files",
        "no_files",
    ],
)
def test_policy_gates_fail_closed(publisher, review_file, api, gate):
    """Withhold approval when a required safety gate is absent."""
    if gate == "no_rules":
        api["rules"] = []
    if gate == "stale_allowed":
        api["rules"][0]["parameters"]["dismiss_stale_reviews_on_push"] = False
    if gate == "zero_approvals":
        api["rules"][0]["parameters"]["required_approving_review_count"] = 0
    if gate == "missing_approval_count":
        del api["rules"][0]["parameters"]["required_approving_review_count"]
    if gate == "no_check_rule":
        api["rules"] = api["rules"][:1]
    if gate == "non_strict_checks":
        api["rules"][1]["parameters"]["strict_required_status_checks_policy"] = False
    if gate == "empty_checks":
        api["rules"][1]["parameters"]["required_status_checks"] = []
    if gate == "truncated_files":
        api["pr"]["changed_files"] = 2
    if gate == "no_files":
        api["files"] = []
        api["pr"]["changed_files"] = 0
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", gate != "disabled")
    assert api["posted"][0]["event"] == "COMMENT"


@pytest.mark.parametrize("change", ["head", "base", "draft", "closed"])
def test_obsolete_or_closed_pr_skipped(publisher, review_file, api, change):
    """Skip artifacts whose PR state no longer matches the review event."""
    if change in {"head", "base"}:
        api["pr"][change]["sha"] = "d" * 40
    elif change == "draft":
        api["pr"]["draft"] = True
    else:
        api["pr"]["state"] = "closed"
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert not api["posted"]


@pytest.mark.parametrize("change_at", [2, 3])
def test_revision_races(publisher, review_file, api, change_at):
    """Skip or revoke approvals when the head changes around posting."""
    api["change_at"] = change_at
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    if change_at == 2:
        assert not api["posted"]
    else:
        assert api["posted"][0]["event"] == "APPROVE"
        assert len(api["dismissed"]) == 1
        assert "/reviews/42/dismissals" in api["dismissed"][0][2]


@pytest.mark.parametrize("retarget_at", [1, 2, "post", 3])
def test_same_sha_base_retarget_invalidates_review(publisher, review_file, api, retarget_at):
    """Detect base retargets even when the commit SHA stays unchanged."""
    api["retarget_at"] = retarget_at
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert api["pr"]["base"]["sha"] == BASE
    if retarget_at in {1, 2}:
        assert not api["posted"]
    else:
        assert api["posted"][0]["event"] == "APPROVE"
        assert len(api["dismissed"]) == 1
        assert "/reviews/42/dismissals" in api["dismissed"][0][2]


def test_invalid_replacement_revokes_only_our_approval(publisher, review_file, api):
    """Revoke our earlier approval before rejecting a bad replacement."""
    api["reviews"] = [
        {"id": 1, "user": {"login": "github-actions[bot]"}, "state": "APPROVED", "body": publisher.MARKER},
        {"id": 2, "user": {"login": "maintainer"}, "state": "APPROVED", "body": publisher.MARKER},
        {"id": 3, "user": {"login": "github-actions[bot]"}, "state": "APPROVED", "body": "Other workflow"},
    ]
    review_file.write_text("not JSON")
    with pytest.raises(ValueError):
        publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert len(api["dismissed"]) == 1
    assert "/reviews/1/dismissals" in api["dismissed"][0][2]
    assert not api["posted"]


def test_model_output_is_only_api_data(publisher, review_file, api):
    """Preserve model text as API data without shell interpolation."""
    text = "$(touch /tmp/should-not-exist) `echo secret` \n::set-output name=approve::true"
    review_file.write_text(json.dumps({"verdict": "incomplete", "summary": text, "findings": []}))
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert text in api["posted"][0]["body"]
    assert api["posted"][0]["event"] == "COMMENT"


def test_retarget_revokes_only_workflow_approvals(publisher, api):
    """Revoke only this workflow's active approvals, even on an ineligible PR."""
    api["pr"]["draft"] = True
    api["pr"]["base"]["ref"] = "release"
    api["reviews"] = [
        {"id": 1, "user": {"login": "github-actions[bot]"}, "state": "APPROVED", "body": publisher.MARKER},
        {"id": 2, "user": {"login": "maintainer"}, "state": "APPROVED", "body": publisher.MARKER},
        {"id": 3, "user": {"login": "other[bot]"}, "state": "APPROVED", "body": publisher.MARKER},
        {"id": 4, "user": {"login": "github-actions[bot]"}, "state": "APPROVED", "body": "Other workflow"},
        {"id": 5, "user": {"login": "github-actions[bot]"}, "state": "DISMISSED", "body": publisher.MARKER},
    ]
    publisher.revoke_approvals("owner/repo", "12")
    assert len(api["dismissed"]) == 1
    assert "/reviews/1/dismissals" in api["dismissed"][0][2]
    assert not api["posted"]
    assert api["reads"] == 0


@pytest.mark.parametrize("operation", ["revoke", "publish"])
@pytest.mark.parametrize("failure", ["list", "dismiss"])
def test_revocation_errors_fail_closed(publisher, review_file, api, monkeypatch, operation, failure):
    """Abort invalidation or replacement publication if revocation cannot finish."""
    api["reviews"] = [
        {"id": 1, "user": {"login": "github-actions[bot]"}, "state": "APPROVED", "body": publisher.MARKER}
    ]
    request = publisher.github_api

    def fail(*args, **kwargs):
        """Simulate a denied or unavailable dismissal API."""
        raise RuntimeError("GitHub unavailable")

    def fail_listing(endpoint, *args, **kwargs):
        """Fail only the review-list read used by approval revocation."""
        if "/reviews?" in endpoint:
            return fail()
        return request(endpoint, *args, **kwargs)

    if failure == "list":
        monkeypatch.setattr(publisher, "github_api", fail_listing)
    else:
        monkeypatch.setattr(publisher.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="GitHub unavailable"):
        if operation == "revoke":
            publisher.revoke_approvals("owner/repo", "12")
        else:
            publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert not api["posted"]


@pytest.fixture
def workflow():
    """Load the actual workflow to exercise its event and dependency expressions."""
    path = Path(__file__).resolve().parents[3] / ".github/workflows/codex-review.yml"
    return yaml.safe_load(path.read_text())


def workflow_expression(expression, context, *, cancelled=False, success=True):
    """Evaluate the boolean subset used by this workflow against event fixtures."""

    def evaluate(node):
        """Interpret context reads, boolean comparisons, and supported status checks."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return context[node.id]
        if isinstance(node, ast.Attribute):
            value = evaluate(node.value)
            assert isinstance(value, dict)
            return value.get(node.attr, {})
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name) and node.func.id in {"cancelled", "success"}
            assert not node.args and not node.keywords
            return cancelled if node.func.id == "cancelled" else success
        if isinstance(node, ast.BoolOp):
            values = (bool(evaluate(value)) for value in node.values)
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not evaluate(node.operand)
        if isinstance(node, ast.Compare):
            assert len(node.ops) == len(node.comparators) == 1
            left, right = evaluate(node.left), evaluate(node.comparators[0])
            if isinstance(node.ops[0], ast.Eq):
                return left == right
            assert isinstance(node.ops[0], ast.NotEq)
            return left != right
        raise AssertionError(f"Unsupported workflow expression: {ast.dump(node)}")

    expression = expression.removeprefix("${{").removesuffix("}}")
    expression = expression.replace("&&", "and").replace("||", "or").replace("== false", "== False")
    expression = re.sub(r"!(?!=)", "not ", expression)
    tree = ast.parse(expression.strip(), mode="eval")
    # Actions adds success() unless the expression contains a status function.
    has_status_function = any(isinstance(node, ast.Call) for node in ast.walk(tree))
    return (has_status_function or success) and evaluate(tree.body)


@pytest.mark.parametrize(
    "action,retarget,actor,draft,enabled,invalidation,invalidates,reviews",
    [
        ("opened", False, "User", False, "true", "skipped", False, True),
        ("opened", False, "Bot", False, "true", "skipped", False, False),
        ("edited", False, "User", False, "true", "skipped", False, False),
        ("edited", True, "User", False, "true", "success", True, True),
        ("edited", True, "Bot", False, "true", "success", True, True),
        ("edited", True, "Bot", True, "false", "success", True, False),
        ("edited", True, "User", False, "true", "failure", True, False),
        ("edited", True, "User", False, "true", "cancelled", True, False),
    ],
)
def test_workflow_retarget_gates(
    workflow, action, retarget, actor, draft, enabled, invalidation, invalidates, reviews
):
    """Retargets invalidate first and unrelated edits never enqueue review jobs."""
    trigger = workflow.get("on", workflow.get(True))
    assert "edited" in trigger["pull_request_target"]["types"]
    context = {
        "github": {
            "event_name": "pull_request_target",
            "run_id": 12345,
            "event": {
                "action": action,
                "changes": {"base": {"ref": {"from": "main"}}} if retarget else {},
                "sender": {"type": actor},
                "pull_request": {"draft": draft, "number": 12},
            },
        },
        "vars": {"CODEX_REVIEW_ENABLED": enabled},
        "needs": {"invalidate": {"result": invalidation}},
    }
    assert bool(workflow_expression(workflow["jobs"]["invalidate"]["if"], context)) == invalidates
    assert workflow["jobs"]["review"]["needs"] == "invalidate"
    assert bool(workflow_expression(workflow["jobs"]["review"]["if"], context)) == reviews
    assert not workflow_expression(workflow["jobs"]["review"]["if"], context, cancelled=True)


@pytest.mark.parametrize("invalidation", ["skipped", "success"])
@pytest.mark.parametrize("review", ["success", "failure", "skipped", "cancelled"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_publish_requires_successful_review_after_invalidation(workflow, invalidation, review, cancelled):
    """A skipped ancestor must not suppress publishing a successful review."""
    publish = workflow["jobs"]["publish"]
    assert publish["needs"] == "review"
    context = {"needs": {"review": {"result": review}}}
    assert bool(
        workflow_expression(
            publish.get("if", "success()"),
            context,
            cancelled=cancelled,
            success=invalidation == review == "success" and not cancelled,
        )
    ) == (review == "success" and not cancelled)


def test_invalidation_is_independent_of_review_cancellation(workflow):
    """New review events cannot cancel a running approval revocation job."""
    assert "concurrency" not in workflow
    jobs = workflow["jobs"]
    assert jobs["invalidate"]["concurrency"]["cancel-in-progress"] is False
    assert jobs["review"]["concurrency"]["cancel-in-progress"] is True
    assert jobs["publish"]["concurrency"]["cancel-in-progress"] is True
    groups = [jobs[name]["concurrency"]["group"] for name in ("invalidate", "review", "publish")]
    assert len(set(groups)) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
