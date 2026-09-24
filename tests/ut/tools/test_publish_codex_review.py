# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression tests for automated PR approval boundaries, without GitHub writes."""

import importlib.util
import json
from pathlib import Path

import pytest

HEAD = "a" * 40
BASE = "b" * 40


@pytest.fixture
def publisher():
    path = Path(__file__).resolve().parents[3] / ".github/scripts/publish_codex_review.py"
    spec = importlib.util.spec_from_file_location("publish_codex_review", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def review_file(tmp_path):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"verdict": "pass", "summary": "Reviewed all changes.", "findings": []}))
    return path


@pytest.fixture
def api(publisher, monkeypatch):
    state = {
        "pr": {
            "state": "open",
            "draft": False,
            "head": {"sha": HEAD},
            "base": {"sha": BASE, "ref": "main"},
            "changed_files": 1,
        },
        "files": [{"filename": "src/example.cpp"}],
        "rules": [{"type": "pull_request", "parameters": {"dismiss_stale_reviews_on_push": True}}],
        "reviews": [],
        "posted": [],
        "dismissed": [],
        "reads": 0,
    }

    def request(endpoint, payload=None, *, paginate=False):
        if payload is not None:
            state["posted"].append(payload)
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
        if state.get("change_at") == state["reads"]:
            state["pr"]["head"]["sha"] = "c" * 40
        return json.loads(json.dumps(state["pr"]))

    def run(command, **kwargs):
        assert command[-4:] == ["--method", "PUT", "--input", "-"]
        state["dismissed"].append(command)

    monkeypatch.setattr(publisher, "github_api", request)
    monkeypatch.setattr(publisher.subprocess, "run", run)
    return state


def test_clean_review_approves_exact_commit(publisher, review_file, api):
    assert publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, True).startswith("Approved")
    assert api["posted"][0]["event"] == "APPROVE"
    assert api["posted"][0]["commit_id"] == HEAD


@pytest.mark.parametrize(
    "verdict,findings",
    [
        ("findings", [{"title": "[P1] Bug", "body": "src/example.cpp:10 has an overflow."}]),
        ("incomplete", []),
    ],
)
def test_nonpassing_review_never_approves(publisher, review_file, api, verdict, findings):
    review_file.write_text(
        json.dumps({"verdict": verdict, "summary": "Needs attention", "findings": findings})
    )
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, True)
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
    review_file.write_text(contents)
    with pytest.raises(ValueError):
        publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, True)
    assert not api["posted"]


def test_symlink_rejected(publisher, review_file, api, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(review_file)
    with pytest.raises(ValueError, match="regular file"):
        publisher.publish(link, "owner/repo", "12", HEAD, BASE, True)
    assert not api["posted"]


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/new.yml",
        ".github/scripts/new.py",
        "AGENTS.md",
        "src/AGENTS.md",
        ".codex/config.toml",
        ".claude/rules/review.md",
        ".agents/skills/policy",
        ".gitmodules",
    ],
)
@pytest.mark.parametrize("rename", [False, True])
def test_policy_changes_require_human_review(publisher, review_file, api, path, rename):
    api["files"] = [{"filename": "renamed.txt", "previous_filename": path} if rename else {"filename": path}]
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, True)
    assert api["posted"][0]["event"] == "COMMENT"


@pytest.mark.parametrize("gate", ["disabled", "no_rules", "stale_allowed", "truncated_files", "no_files"])
def test_policy_gates_fail_closed(publisher, review_file, api, gate):
    if gate == "no_rules":
        api["rules"] = []
    if gate == "stale_allowed":
        api["rules"][0]["parameters"]["dismiss_stale_reviews_on_push"] = False
    if gate == "truncated_files":
        api["pr"]["changed_files"] = 2
    if gate == "no_files":
        api["files"] = []
        api["pr"]["changed_files"] = 0
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, gate != "disabled")
    assert api["posted"][0]["event"] == "COMMENT"


@pytest.mark.parametrize("change", ["head", "base", "draft", "closed"])
def test_obsolete_or_closed_pr_skipped(publisher, review_file, api, change):
    if change in {"head", "base"}:
        api["pr"][change]["sha"] = "d" * 40
    elif change == "draft":
        api["pr"]["draft"] = True
    else:
        api["pr"]["state"] = "closed"
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, True)
    assert not api["posted"]


@pytest.mark.parametrize("change_at", [2, 3])
def test_revision_races(publisher, review_file, api, change_at):
    api["change_at"] = change_at
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, True)
    if change_at == 2:
        assert not api["posted"]
    else:
        assert api["posted"][0]["event"] == "APPROVE"
        assert len(api["dismissed"]) == 1
        assert "/reviews/42/dismissals" in api["dismissed"][0][2]


def test_invalid_replacement_revokes_only_our_approval(publisher, review_file, api):
    api["reviews"] = [
        {"id": 1, "user": {"login": "github-actions[bot]"}, "state": "APPROVED", "body": publisher.MARKER},
        {"id": 2, "user": {"login": "maintainer"}, "state": "APPROVED", "body": publisher.MARKER},
        {"id": 3, "user": {"login": "github-actions[bot]"}, "state": "APPROVED", "body": "Other workflow"},
    ]
    review_file.write_text("not JSON")
    with pytest.raises(ValueError):
        publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, True)
    assert len(api["dismissed"]) == 1
    assert "/reviews/1/dismissals" in api["dismissed"][0][2]
    assert not api["posted"]


def test_model_output_is_only_api_data(publisher, review_file, api):
    text = "$(touch /tmp/should-not-exist) `echo secret` \n::set-output name=approve::true"
    review_file.write_text(json.dumps({"verdict": "incomplete", "summary": text, "findings": []}))
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, True)
    assert text in api["posted"][0]["body"]
    assert api["posted"][0]["event"] == "COMMENT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
