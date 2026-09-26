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
        "reviews": [],
        "comments": [],
        "posted": [],
        "dismissed": [],
        "reads": 0,
    }

    def request(endpoint, payload=None, *, paginate=False):
        """Return API fixtures and inject the requested publication race."""
        if payload is not None:
            assert endpoint.endswith("/reviews"), "Review publishing must not merge pull requests"
            state["posted"].append(payload)
            if state.get("retarget_at") == "post":
                state["pr"]["base"]["ref"] = "release"
            return {"id": 42}
        if "/files?" in endpoint:
            assert paginate
            return state["files"]
        if "/rules/" in endpoint:
            raise AssertionError("Review approval must not depend on branch protection rules")
        if "/reviews?" in endpoint:
            return state["reviews"]
        if "/comments?" in endpoint:
            assert paginate
            return state["comments"]
        if "/compare/" in endpoint:
            base_sha = endpoint.split("/compare/", 1)[1].split("...", 1)[0]
            if state.get("invalid_merge_base"):
                return {"merge_base_commit": {"sha": "invalid"}}
            changed = state.get("merge_base_changed") and base_sha != BASE
            return {"merge_base_commit": {"sha": ("f" if changed else "e") * 40}}
        state["reads"] += 1
        if state.get("retarget_at") == state["reads"]:
            state["pr"]["base"]["ref"] = "release"
        if state.get("change_at") == state["reads"]:
            state["pr"]["head"]["sha"] = "c" * 40
        if state.get("base_change_at") == state["reads"]:
            state["pr"]["base"]["sha"] = "d" * 40
        return json.loads(json.dumps(state["pr"]))

    def run(command, **kwargs):
        """Record approval dismissals instead of invoking GitHub."""
        assert command[-4:] == ["--method", "PUT", "--input", "-"]
        state["dismissed"].append(command)

    monkeypatch.setattr(publisher, "github_api", request)
    monkeypatch.setattr(publisher.subprocess, "run", run)
    return state


@pytest.fixture
def located_review(review_file, api):
    """Supply a replacement hunk with unequal old and new line numbers."""
    api["files"] = [
        {
            "filename": "src/example.cpp",
            "patch": "@@ -10,3 +20,3 @@\n context\n-old\n+new\n tail",
        }
    ]

    def write(location, **extra):
        finding = {"title": "[P1] Bug", "body": "This change breaks empty inputs.", "location": location}
        finding.update(extra)
        review_file.write_text(
            json.dumps({"verdict": "findings", "summary": "Needs attention", "findings": [finding]})
        )
        return finding

    return write


@pytest.mark.parametrize("side,line", [("RIGHT", 21), ("LEFT", 11), ("RIGHT", 20)])
def test_inline_findings(publisher, review_file, api, located_review, side, line):
    """Anchor added, deleted and context lines to the examined commit."""
    location = {"path": "src/example.cpp", "side": side, "line": line}
    finding = located_review(location)
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    posted = api["posted"][0]
    assert posted["commit_id"] == HEAD and posted["event"] == "COMMENT"
    assert posted["comments"] == [
        {**location, "body": f"{publisher.MARKER}\n### {finding['title']}\n\n{finding['body']}"}
    ]
    assert finding["body"] not in posted["body"]


@pytest.mark.parametrize(
    "location",
    [
        None,
        {},
        "src/example.cpp:21",
        {"path": "../secret", "line": 21, "side": "RIGHT"},
        {"path": "/src/example.cpp", "line": 21, "side": "RIGHT"},
        {"path": "src/example.cpp", "line": True, "side": "RIGHT"},
        {"path": "src/example.cpp", "line": 0, "side": "RIGHT"},
        {"path": "src/example.cpp", "line": 21, "side": "WRONG"},
        {"path": "src/example.cpp", "line": 11, "side": "RIGHT"},
        {"path": "src/example.cpp", "line": 500, "side": "RIGHT"},
        {"path": "src/unchanged.cpp", "line": 21, "side": "RIGHT"},
    ],
)
def test_unanchored_findings_survive(publisher, review_file, api, located_review, location):
    """Invalid, absent and out-of-diff locations never hide actionable findings."""
    finding = located_review(location)
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    posted = api["posted"][0]
    assert posted["event"] == "COMMENT"
    assert not posted.get("comments")
    assert finding["body"] in posted["body"]
    if isinstance(location, dict) and location.get("line") == 500:
        assert f"/blob/{HEAD}/src/example.cpp#L500" in posted["body"]


def test_deleted_fallback_links_merge_base_before_rename(publisher, review_file, api, located_review):
    """LEFT coordinates refer to the merge base, not the potentially advanced base tip."""
    located_review({"path": "src/example.cpp", "line": 500, "side": "LEFT"})
    api["files"][0]["previous_filename"] = "src/old name.cpp"
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert f"/blob/{'e' * 40}/src/old%20name.cpp#L500" in api["posted"][0]["body"]


@pytest.mark.parametrize("location", [None, {}, {"path": "../invalid", "line": 1, "side": "RIGHT"}])
def test_unanchored_review_needs_no_metadata(
    publisher, review_file, api, located_review, monkeypatch, location
):
    """Summary-only reviews publish even when location metadata APIs are unavailable."""
    finding = located_review(location)
    request = publisher.github_api

    def fail_metadata(endpoint, *args, **kwargs):
        assert "/files?" not in endpoint and "/comments?" not in endpoint
        return request(endpoint, *args, **kwargs)

    monkeypatch.setattr(publisher, "github_api", fail_metadata)
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert finding["body"] in api["posted"][0]["body"]
    assert api["posted"][0]["event"] == "COMMENT"


@pytest.mark.parametrize("failure", ["request", "json", "missing"])
def test_merge_base_lookup_failure_keeps_findings(
    publisher, review_file, api, located_review, monkeypatch, failure
):
    """A failed diff check keeps findings visible without approving."""
    finding = located_review({"path": "src/example.cpp", "line": 500, "side": "LEFT"})
    request = publisher.github_api

    def fail_compare(endpoint, *args, **kwargs):
        if "/compare/" in endpoint:
            if failure == "request":
                raise publisher.subprocess.CalledProcessError(1, ["gh", "api"])
            if failure == "json":
                raise json.JSONDecodeError("Invalid response", "", 0)
            return {}
        return request(endpoint, *args, **kwargs)

    monkeypatch.setattr(publisher, "github_api", fail_compare)
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    posted = api["posted"][0]
    assert posted["event"] == "COMMENT"
    assert finding["body"] in posted["body"]
    assert "Cannot verify that the reviewed PR diff is current" in posted["body"]
    assert "/blob/" not in posted["body"]
    assert not posted.get("comments")


@pytest.mark.parametrize("endpoint_part", ["/files?", "/comments?"])
@pytest.mark.parametrize("failure", ["request", "json"])
def test_location_metadata_failure_keeps_findings(
    publisher, review_file, api, located_review, monkeypatch, endpoint_part, failure
):
    """Optional location and deduplication reads cannot suppress a valid review."""
    finding = located_review({"path": "src/example.cpp", "line": 21, "side": "RIGHT"})
    request = publisher.github_api

    def fail_metadata(endpoint, *args, **kwargs):
        if endpoint_part in endpoint:
            if failure == "json":
                raise json.JSONDecodeError("Invalid response", "", 0)
            raise publisher.subprocess.CalledProcessError(1, ["gh", "api"])
        return request(endpoint, *args, **kwargs)

    monkeypatch.setattr(publisher, "github_api", fail_metadata)
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    posted = api["posted"][0]
    assert finding["body"] in posted["body"]
    assert "src/example.cpp:21 (RIGHT)" in posted["body"]
    assert not posted.get("comments") and posted["event"] == "COMMENT"


@pytest.mark.parametrize("patch", ["", "@@ -10,3 +20,3 @@\n context\n-old", "not a patch"])
def test_unusable_patch_falls_back(publisher, review_file, api, located_review, patch):
    """Missing and truncated patches cannot create unverified inline comments."""
    finding = located_review({"path": "src/example.cpp", "line": 20, "side": "RIGHT"})
    api["files"][0]["patch"] = patch
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert not api["posted"][0].get("comments")
    assert finding["body"] in api["posted"][0]["body"]


@pytest.mark.parametrize(
    "patch,expected",
    [
        ("@@ -0,0 +1,2 @@\n+a\n+b", {("RIGHT", 1), ("RIGHT", 2)}),
        ("@@ -1 +0,0 @@\n-a\n\\ No newline at end of file", {("LEFT", 1)}),
        (
            "@@ -1 +1 @@\n-a\n+b\n@@ -20 +30 @@\n-c\n+d",
            {("LEFT", 1), ("RIGHT", 1), ("LEFT", 20), ("RIGHT", 30)},
        ),
    ],
)
def test_patch_coordinates(publisher, patch, expected):
    """Handle new/deleted files, omitted counts, multiple hunks and newline markers."""
    assert publisher.patch_lines(patch) == expected


@pytest.mark.parametrize(
    "previous_head,original_head,author,duplicate",
    [
        (HEAD, HEAD, "github-actions[bot]", True),
        (BASE, BASE, "github-actions[bot]", False),
        (HEAD, HEAD, "human", False),
        (HEAD, BASE, "github-actions[bot]", False),
        (HEAD, None, "github-actions[bot]", False),
    ],
)
def test_inline_rerun_deduplication(
    publisher, review_file, api, located_review, previous_head, original_head, author, duplicate
):
    """Only identical workflow comments on the same commit suppress a new thread."""
    located_review({"path": "src/example.cpp", "line": 21, "side": "RIGHT"})
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    comment = api["posted"][0]["comments"][0]
    api["comments"] = [
        {
            **comment,
            "commit_id": previous_head,
            "original_commit_id": original_head,
            "user": {"login": author},
        }
    ]
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert bool(api["posted"][1].get("comments")) is not duplicate
    if duplicate:
        assert "src/example.cpp:21 (RIGHT)" in api["posted"][1]["body"]


def test_inline_result_skipped_after_push(publisher, review_file, api, located_review):
    """Recheck revision after fetching anchors and comments, before publishing."""
    located_review({"path": "src/example.cpp", "line": 21, "side": "RIGHT"})
    api["change_at"] = 2
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert not api["posted"]


@pytest.mark.parametrize("author_metadata", [{"user": None}, {}])
def test_authorless_prior_comment_does_not_block_publication(
    publisher, review_file, api, located_review, author_metadata
):
    """Deleted or unavailable comment authors cannot break the deduplication scan."""
    located_review({"path": "src/example.cpp", "line": 21, "side": "RIGHT"})
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    comment = api["posted"][0]["comments"][0]
    api["comments"] = [{**comment, "original_commit_id": HEAD, **author_metadata}]
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert api["posted"][1]["event"] == "COMMENT"
    assert api["posted"][1]["comments"] == [comment]


def test_mixed_findings_and_batch_limit(publisher, review_file, api, located_review):
    """A full inline batch must retain every remaining finding in the summary."""
    finding = located_review({"path": "src/example.cpp", "line": 21, "side": "RIGHT"})
    findings = [{**finding, "title": f"Bug {index}"} for index in range(101)]
    findings.append({"title": "General issue", "body": "Unanchored issue", "location": None})
    review_file.write_text(json.dumps({"verdict": "findings", "summary": "Issues", "findings": findings}))
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    posted = api["posted"][0]
    assert len(posted["comments"]) == 100
    assert "Bug 100" in posted["body"] and "Unanchored issue" in posted["body"]


def test_inline_size_budget(publisher, review_file, api, located_review):
    """Include inline comment text in the existing rendered-output byte budget."""
    located_review({"path": "src/example.cpp", "line": 21, "side": "RIGHT"}, body="x" * 30000)
    # The artifact is small enough to load; the rendered review exceeds this budget.
    publisher.MAX_BYTES = review_file.stat().st_size + 1
    with pytest.raises(ValueError, match="Rendered review"):
        publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert not api["posted"]


def test_clean_review_approves_exact_commit_without_branch_rules(publisher, review_file, api):
    """Bind a clean review to the examined commit without reading merge rules."""
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
        "truncated_files",
        "no_files",
    ],
)
def test_review_gates_fail_closed(publisher, review_file, api, gate):
    """Withhold approval when a required review input is absent."""
    if gate == "truncated_files":
        api["pr"]["changed_files"] = 2
    if gate == "no_files":
        api["files"] = []
        api["pr"]["changed_files"] = 0
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", gate != "disabled")
    assert api["posted"][0]["event"] == "COMMENT"


@pytest.mark.parametrize("change", ["head", "draft", "closed"])
def test_obsolete_or_closed_pr_skipped(publisher, review_file, api, change):
    """Skip artifacts whose PR state no longer matches the review event."""
    if change == "head":
        api["pr"][change]["sha"] = "d" * 40
    elif change == "draft":
        api["pr"]["draft"] = True
    else:
        api["pr"]["state"] = "closed"
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert not api["posted"]


@pytest.mark.parametrize("base_change_at", [1, 2, 3])
def test_base_advance_does_not_invalidate_approval(publisher, review_file, api, base_change_at):
    """A base-branch commit must not invalidate a review of the same PR head."""
    api["base_change_at"] = base_change_at
    assert publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True).startswith("Approved")
    assert api["pr"]["base"]["sha"] != BASE
    assert api["posted"][0]["event"] == "APPROVE"
    assert not api["dismissed"]


@pytest.mark.parametrize("base_change_at", [1, 2, 3])
def test_merge_base_change_blocks_stale_approval(publisher, review_file, api, base_change_at):
    """Reject a base rewrite that changes the diff despite an unchanged PR head."""
    api["base_change_at"] = base_change_at
    api["merge_base_changed"] = True
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    if base_change_at == 3:
        assert api["posted"][0]["event"] == "APPROVE"
        assert len(api["dismissed"]) == 1
        assert "/reviews/42/dismissals" in api["dismissed"][0][2]
    else:
        assert api["posted"][0]["event"] == "COMMENT"
        assert "Cannot verify that the reviewed PR diff is current" in api["posted"][0]["body"]


def test_invalid_merge_base_fails_closed(publisher, review_file, api):
    """Do not publish an approval from an invalid GitHub compare response."""
    api["invalid_merge_base"] = True
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert api["posted"][0]["event"] == "COMMENT"


@pytest.mark.parametrize("failure_at", [3, 4])
def test_compare_failure_around_publication(publisher, review_file, api, monkeypatch, failure_at):
    """Comment before posting or dismiss approval if diff verification fails later."""
    request = publisher.github_api
    compare_reads = 0

    def fail_late_compare(endpoint, *args, **kwargs):
        nonlocal compare_reads
        if "/compare/" in endpoint:
            compare_reads += 1
            if compare_reads == failure_at:
                raise publisher.subprocess.CalledProcessError(1, ["gh", "api"])
        return request(endpoint, *args, **kwargs)

    monkeypatch.setattr(publisher, "github_api", fail_late_compare)
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    if failure_at == 3:
        assert api["posted"][0]["event"] == "COMMENT"
        assert not api["dismissed"]
    else:
        assert api["posted"][0]["event"] == "APPROVE"
        assert len(api["dismissed"]) == 1
        assert "/reviews/42/dismissals" in api["dismissed"][0][2]


def test_prepost_compare_failure_moves_findings_to_summary(
    publisher, review_file, api, located_review, monkeypatch
):
    """Avoid inline anchors when the reviewed diff cannot be verified."""
    finding = located_review({"path": "src/example.cpp", "line": 21, "side": "RIGHT"})
    request = publisher.github_api
    compare_reads = 0

    def fail_prepost_compare(endpoint, *args, **kwargs):
        nonlocal compare_reads
        if "/compare/" in endpoint:
            compare_reads += 1
            if compare_reads == 3:
                raise publisher.subprocess.CalledProcessError(1, ["gh", "api"])
        return request(endpoint, *args, **kwargs)

    monkeypatch.setattr(publisher, "github_api", fail_prepost_compare)
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    posted = api["posted"][0]
    assert posted["event"] == "COMMENT"
    assert finding["body"] in posted["body"]
    assert not posted.get("comments")


def test_post_publication_pr_read_failure_dismisses_approval(publisher, review_file, api, monkeypatch):
    """Dismiss an approval if GitHub cannot confirm the PR after posting."""
    request = publisher.github_api
    pr_reads = 0

    def fail_final_read(endpoint, *args, **kwargs):
        nonlocal pr_reads
        if endpoint == "repos/owner/repo/pulls/12":
            pr_reads += 1
            if pr_reads == 3:
                raise publisher.subprocess.CalledProcessError(1, ["gh", "api"])
        return request(endpoint, *args, **kwargs)

    monkeypatch.setattr(publisher, "github_api", fail_final_read)
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert api["posted"][0]["event"] == "APPROVE"
    assert len(api["dismissed"]) == 1


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

    if expression.startswith("${{"):
        expression = expression[3:]
    if expression.endswith("}}"):
        expression = expression[:-2]
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
    assert workflow["jobs"]["prepare"]["needs"] == "invalidate"
    assert bool(workflow_expression(workflow["jobs"]["prepare"]["if"], context)) == reviews
    assert not workflow_expression(workflow["jobs"]["prepare"]["if"], context, cancelled=True)


@pytest.mark.parametrize("invalidation", ["skipped", "success"])
@pytest.mark.parametrize("review", ["success", "failure", "skipped", "cancelled"])
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("completed", ["true", ""])
def test_publish_requires_successful_review_after_invalidation(
    workflow, invalidation, review, cancelled, completed
):
    """A skipped ancestor must not suppress publishing a successful review."""
    publish = workflow["jobs"]["publish"]
    assert publish["needs"] == "review"
    context = {"needs": {"review": {"result": review, "outputs": {"completed": completed}}}}
    assert bool(
        workflow_expression(
            publish.get("if", "success()"),
            context,
            cancelled=cancelled,
            success=invalidation == review == "success" and not cancelled,
        )
    ) == (review == "success" and not cancelled and completed == "true")


def test_invalidation_is_independent_of_review_cancellation(workflow):
    """New review events cannot cancel a running approval revocation job."""
    assert "concurrency" not in workflow
    jobs = workflow["jobs"]
    assert jobs["invalidate"]["concurrency"]["cancel-in-progress"] is False
    assert jobs["review"]["concurrency"]["cancel-in-progress"] is True
    assert jobs["publish"]["concurrency"]["cancel-in-progress"] is True
    groups = [jobs[name]["concurrency"]["group"] for name in ("invalidate", "review", "publish")]
    assert len(set(groups)) == 3


@pytest.mark.parametrize(
    "script_name",
    ["publish_codex_review", "prepare_codex_review", "manage_codex_review_session", "perf_guard"],
)
def test_ci_helpers_import_on_supported_python(monkeypatch, script_name):
    """Exercise actual imports on the CI matrix, including evaluated type annotations."""
    scripts = Path(__file__).resolve().parents[3] / ".github/scripts"
    path = scripts / f"{script_name}.py"
    ast.parse(path.read_text(), filename=str(path), feature_version=8)
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(script_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def test_review_workflow_has_no_merge_permission(workflow):
    """Review publishing can approve but cannot write repository contents."""
    jobs = workflow["jobs"]
    assert jobs["publish"]["permissions"]["pull-requests"] == "write"
    assert all(job["permissions"].get("contents") != "write" for job in jobs.values())


@pytest.mark.parametrize(
    "reference_author,root,marked,links",
    [
        ("github-actions[bot]", True, True, True),
        ("maintainer", True, True, False),
        ("github-actions[bot]", False, True, False),
        ("github-actions[bot]", True, False, False),
    ],
)
def test_repeat_finding_links_existing_thread(
    publisher, review_file, api, located_review, reference_author, root, marked, links
):
    """Reworded cross-commit findings reuse only verified workflow-owned root comments."""
    located_review({"path": "src/example.cpp", "line": 21, "side": "RIGHT"}, existing_comment_id=123)
    api["comments"] = [
        {
            "id": 123,
            "user": {"login": reference_author},
            "body": publisher.MARKER + "previous wording" if marked else "unrelated review",
            "in_reply_to_id": None if root else 12,
            "path": "src/example.cpp",
            "line": 10,
            "side": "RIGHT",
            "original_commit_id": "c" * 40,
        }
    ]
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    posted = api["posted"][0]
    assert posted["event"] == "COMMENT"
    assert ("#discussion_r123" in posted["body"]) == links
    assert bool(posted.get("comments")) != links


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
def test_repeat_finding_keeps_current_location(publisher, review_file, api, located_review, side):
    """An old thread link must not replace the current finding's reported coordinates."""
    located_review({"path": "src/example.cpp", "line": 21, "side": side}, existing_comment_id=123)
    api["comments"] = [
        {
            "id": 123,
            "user": {"login": "github-actions[bot]"},
            "body": publisher.MARKER + "previous finding",
            "path": "src/example.cpp",
            "line": None,
            "original_line": 10,
            "side": "RIGHT",
            "original_commit_id": "c" * 40,
        }
    ]
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    posted = api["posted"][0]
    assert posted["event"] == "COMMENT"
    assert f"Reported location: src/example.cpp:21 ({side})." in posted["body"]
    assert "#discussion_r123" in posted["body"]
    assert not posted.get("comments")


@pytest.mark.parametrize(
    "previous_path,status,rename_source,location,links",
    [
        ("src/example.cpp", "modified", None, True, True),
        ("src/other.cpp", "modified", None, True, False),
        ("src/old.cpp", "renamed", "src/old.cpp", True, True),
        ("src/old.cpp", "modified", "src/old.cpp", True, False),
        ("src/other.cpp", "renamed", "src/old.cpp", True, False),
        ("src/new.cpp", "added", "reused", True, False),
        (None, "modified", None, True, False),
        ("src/example.cpp", "modified", None, False, False),
    ],
)
def test_thread_reference_requires_matching_file(
    publisher, review_file, api, located_review, previous_path, status, rename_source, location, links
):
    """Only same-file or explicit current-PR rename references can suppress a new thread."""
    located_review(
        {"path": "src/example.cpp", "line": 21, "side": "RIGHT"} if location else None,
        existing_comment_id=123,
    )
    api["files"][0].update(status=status)
    if rename_source == "reused":
        api["files"].append(
            {"filename": "src/new.cpp", "status": "renamed", "previous_filename": "src/example.cpp"}
        )
    elif rename_source:
        api["files"][0]["previous_filename"] = rename_source
    api["comments"] = [
        {
            "id": 123,
            "user": {"login": "github-actions[bot]"},
            "body": publisher.MARKER + "old finding",
            "path": previous_path,
            "line": 9,
            "side": "LEFT",
            "original_commit_id": "c" * 40,
        }
    ]
    publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    posted = api["posted"][0]
    assert posted["event"] == "COMMENT"
    assert ("#discussion_r123" in posted["body"]) == links
    assert bool(posted.get("comments")) == (location and not links)
    if not location:
        assert "## Codex Review" in posted["body"]


@pytest.mark.parametrize("reference", [True, 0, -1, "123"])
def test_invalid_thread_reference_rejected(publisher, review_file, api, located_review, reference):
    """Model-supplied references are data, never an unchecked URL or command."""
    located_review(None, existing_comment_id=reference)
    with pytest.raises(ValueError, match="comment ID"):
        publisher.publish(review_file, "owner/repo", "12", HEAD, BASE, "main", True)
    assert not api["posted"]


@pytest.mark.parametrize("ready,result", [("true", "success"), ("", "success"), ("true", "failure")])
def test_review_requires_authorized_context(workflow, ready, result):
    context = {"needs": {"prepare": {"result": result, "outputs": {"ready": ready}}}}
    assert bool(workflow_expression(workflow["jobs"]["review"]["if"], context)) == (
        ready == "true" and result == "success"
    )
    assert not workflow_expression(workflow["jobs"]["review"]["if"], context, cancelled=True)


def test_comment_trigger_and_session_isolation(workflow):
    """Only trusted default-branch comment events can enter the persistent review sandbox."""
    trigger = workflow.get("on", workflow.get(True))
    assert trigger["issue_comment"]["types"] == ["created"]
    assert "pull_request_review_comment" not in trigger
    jobs = workflow["jobs"]
    for name in ("prepare", "publish", "invalidate", "cleanup-sessions"):
        checkout = next(step for step in jobs[name]["steps"] if "checkout@" in step.get("uses", ""))
        assert checkout["with"]["ref"] == "${{ github.workflow_sha }}"
    review = next(
        step["run"] for step in jobs["review"]["steps"] if step["name"] == "Review in an isolated container"
    )
    assert "pypto-codex-sessions-${REPOSITORY_ID}-${PR_NUMBER}" in review
    assert "dst=/codex-home/sessions" in review and "dst=/codex-state" in review
    assert "--tmpfs /codex-home:" in review
    assert 'resume_args=(resume "$previous_session")' in review
    assert 'exec "${resume_args[@]}" --json --output-schema' in review
    assert "dst=/review-discussion.json,readonly" in review
    assert "GH_TOKEN" not in review
    assert jobs["prepare"]["permissions"]["pull-requests"] == "read"
    assert jobs["review"]["permissions"]["pull-requests"] == "read"
    assert 'git show "$TRUSTED_WORKFLOW_SHA:.github/scripts/manage_codex_review_session.py"' in review
    assert 'git show "$TRUSTED_WORKFLOW_SHA:.github/scripts/publish_codex_review.py"' in review
    assert review.index("flock --exclusive 9") < review.index("session_status=$(python")
    assert review.index("session_status=$(python") < review.index("docker run")
    upload = next(step for step in jobs["review"]["steps"] if step["name"] == "Upload review result")
    assert upload["if"] == "steps.codex.outputs.completed == 'true'"


@pytest.mark.parametrize("enabled", ["true", "false"])
@pytest.mark.parametrize("draft", [True, False])
@pytest.mark.parametrize("sender", ["Bot", "User"])
@pytest.mark.parametrize("merged", [True, False])
def test_close_cleanup_is_independent_of_review_eligibility(workflow, enabled, draft, sender, merged):
    """Every close reaches trusted cleanup, with no model or active-review cancellation."""
    context = {
        "vars": {"CODEX_REVIEW_ENABLED": enabled},
        "github": {
            "event_name": "pull_request_target",
            "event": {
                "action": "closed",
                "sender": {"type": sender},
                "pull_request": {"draft": draft, "merged": merged},
            },
        },
        "needs": {"invalidate": {"result": "skipped"}},
    }
    cleanup = workflow["jobs"]["cleanup-sessions"]
    assert workflow_expression(cleanup["if"], context)
    assert not workflow_expression(workflow["jobs"]["prepare"]["if"], context)
    assert "concurrency" not in cleanup
    run = cleanup["steps"][-1]["run"]
    assert run.index("flock --exclusive 9") < run.index("manage_codex_review_session.py cleanup")
    assert "docker run" not in run


@pytest.mark.parametrize(
    "failure", [None, "missing_id", "duplicate_id", "bad_id", "missing_completion", "failed", "bad_json"]
)
def test_session_checkpoint_requires_valid_completed_review(publisher, tmp_path, failure):
    """Select the root thread ID and final answer, never a tool result or failed turn."""
    session_id = "12345678-1234-1234-1234-123456789abc"
    result = {"verdict": "pass", "summary": "Reviewed", "findings": []}
    events = [
        {"type": "thread.started", "thread_id": session_id},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Working..."}},
        {
            "type": "item.completed",
            "item": {"type": "command_execution", "aggregated_output": "not review JSON"},
        },
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(result)}},
        {"type": "turn.completed"},
    ]
    if failure == "missing_id":
        events.pop(0)
    elif failure == "duplicate_id":
        events.insert(0, events[0])
    elif failure == "bad_id":
        events[0]["thread_id"] = "$(shell text)"
    elif failure == "missing_completion":
        events.pop()
    elif failure == "failed":
        events.append({"type": "turn.failed"})
    elif failure == "bad_json":
        events[-2]["item"]["text"] = "{}"
    source, output = tmp_path / "events.jsonl", tmp_path / "review.json"
    source.write_text("".join(json.dumps(event) + "\n" for event in events))
    if failure:
        with pytest.raises(ValueError):
            publisher.extract_session(source, output)
    else:
        assert publisher.extract_session(source, output) == session_id
        assert json.loads(output.read_text()) == result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
