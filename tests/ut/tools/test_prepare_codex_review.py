# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test trigger authorization and lossless GitHub discussion collection without network writes."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def collector(monkeypatch):
    scripts = Path(__file__).resolve().parents[3] / ".github/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("prepare_codex_review", scripts / "prepare_codex_review.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pr():
    return {
        "number": 2911,
        "state": "open",
        "draft": False,
        "head": {"sha": "a" * 40},
        "base": {"sha": "b" * 40, "ref": "main", "repo": {"full_name": "owner/repo"}},
        "user": {"login": "author"},
        "title": "Roundtrip",
        "body": "Preserve declared IR types",
    }


@pytest.fixture
def comment_event():
    return {
        "action": "created",
        "issue": {"number": 2911, "pull_request": {"url": "url"}},
        "comment": {"user": {"login": "author", "type": "User"}, "body": "@pypto-codex review"},
    }


@pytest.mark.parametrize(
    "body,expected",
    [
        ("@pypto-codex review", True),
        ("Explanation first.\n\n@pypto-codex review\n", True),
        ("@PYPTO-CODEX review", True),
        ("Please @pypto-codex review", False),
        ("> @pypto-codex review", False),
        ("    @pypto-codex review", False),
        ("\t@pypto-codex review", False),
        ("  \t@pypto-codex review", False),
        ("Example:\n\n    @pypto-codex review\n", False),
        ("    Example\n\n@pypto-codex review", True),
        ("   @pypto-codex review", True),
        ("```\n    ```\n@pypto-codex review\n```", False),
        ("```\n@pypto-codex review\n```", False),
        ("~~~text\n@pypto-codex review\n~~~", False),
        ("```\nexample\n```\n@pypto-codex review", True),
        ("@pypto-codex reviewer", False),
        ("@codex review", False),
        ("@pypto-codex review; touch /tmp/bad", False),
    ],
)
def test_explicit_command(collector, body, expected):
    assert collector.requests_review(body) is expected


@pytest.mark.parametrize(
    "opener,closer,expected",
    [
        ("```", "```python", False),
        ("~~~", "~~~text", False),
        ("```", "``` trailing text", False),
        ("```", "```~", False),
        ("````", "```", False),
        ("```", "~~~", False),
        ("```python", "```", True),
        ("~~~text", "~~~", True),
        ("```", "````", True),
        ("```", "``` \t", True),
        ("~~~", "  ~~~~ \t", True),
    ],
)
def test_command_requires_bare_closing_fence(collector, opener, closer, expected):
    """Fence-like content must not expose a command that is still inside a block."""
    assert collector.requests_review(f"{opener}\nexample\n{closer}\n@pypto-codex review") is expected


@pytest.mark.parametrize(
    "author,permission,expected",
    [
        ("author", None, True),
        ("maintainer", "write", True),
        ("maintainer", "maintain", True),
        ("maintainer", "admin", True),
        ("outsider", "read", False),
        ("triager", "triage", False),
        ("outsider", "none", False),
    ],
)
def test_comment_authorization(collector, pr, comment_event, monkeypatch, author, permission, expected):
    comment_event["comment"]["user"]["login"] = author
    calls = []

    def api(endpoint):
        calls.append(endpoint)
        return {"permission": permission} if endpoint.endswith("/permission") else pr

    monkeypatch.setattr(collector, "github_api", api)
    assert (collector.review_target("issue_comment", comment_event, "owner/repo") is not None) is expected
    assert len(calls) == (1 if author == "author" else 2)


@pytest.mark.parametrize(
    "stderr,denied",
    [
        ("gh: Not Found (HTTP 404)\n", True),
        ("gh: Bad credentials (HTTP 401)\n", False),
        ("gh: Resource not accessible (HTTP 403)\n", False),
        ("gh: API rate limit exceeded (HTTP 429)\n", False),
        ("gh: Internal Server Error (HTTP 500)\n", False),
        ("Get https://api.github.com: unexpected EOF", False),
        ("404", False),
        (None, False),
    ],
)
def test_permission_lookup_errors(collector, pr, comment_event, monkeypatch, stderr, denied):
    """Only an explicit permission endpoint 404 skips an unauthorized command."""
    comment_event["comment"]["user"]["login"] = "outsider"
    error = subprocess.CalledProcessError(1, ["gh", "api"], stderr=stderr)

    def api(endpoint):
        if endpoint.endswith("/permission"):
            raise error
        return pr

    monkeypatch.setattr(collector, "github_api", api)
    if denied:
        assert collector.review_target("issue_comment", comment_event, "owner/repo") is None
    else:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            collector.review_target("issue_comment", comment_event, "owner/repo")
        assert caught.value is error


@pytest.mark.parametrize("kind", ["bot", "ordinary", "edited", "issue", "inline"])
def test_irrelevant_events_do_not_read_pr(collector, comment_event, monkeypatch, kind):
    if kind == "bot":
        comment_event["comment"]["user"]["type"] = "Bot"
    elif kind == "ordinary":
        comment_event["comment"]["body"] = "Thanks"
    elif kind == "edited":
        comment_event["action"] = "edited"
    elif kind == "issue":
        del comment_event["issue"]["pull_request"]
    monkeypatch.setattr(collector, "github_api", lambda *args: pytest.fail("Unexpected GitHub request"))
    name = "pull_request_review_comment" if kind == "inline" else "issue_comment"
    assert collector.review_target(name, comment_event, "owner/repo") is None


@pytest.mark.parametrize("kind", ["draft", "closed", "foreign_repo", "stale_head", "retargeted"])
def test_nonreviewable_or_obsolete_pr(collector, pr, monkeypatch, kind):
    event = {"action": "synchronize", "sender": {"type": "User"}, "pull_request": json.loads(json.dumps(pr))}
    if kind == "draft":
        pr["draft"] = True
    elif kind == "closed":
        pr["state"] = "closed"
    elif kind == "foreign_repo":
        pr["base"]["repo"]["full_name"] = "other/repo"
    elif kind == "stale_head":
        pr["head"]["sha"] = "c" * 40
    else:
        pr["base"]["ref"] = "release"
    monkeypatch.setattr(collector, "github_api", lambda *args: pr)
    assert collector.review_target("pull_request_target", event, "owner/repo") is None


def test_thread_and_reply_pagination(collector, monkeypatch):
    """Retain corrections after both outer thread and inner reply pagination boundaries."""
    calls = []

    def api(endpoint, payload):
        assert endpoint == "graphql"
        variables = payload["variables"]
        cursor = variables["cursor"]
        calls.append(variables)
        if "number" in variables:
            value = {
                "nodes": [{"id": "thread2" if cursor else "thread1", "isResolved": bool(cursor)}],
                "pageInfo": {"hasNextPage": cursor is None, "endCursor": "next-thread"},
            }
            return {"data": {"repository": {"pullRequest": {"reviewThreads": value}}}}
        value = {
            "nodes": [
                {
                    "body": "Expert correction" if cursor else "Initial finding",
                    "databaseId": 2 if cursor else 1,
                }
            ],
            "pageInfo": {"hasNextPage": cursor is None, "endCursor": "next-comment"},
        }
        return {"data": {"node": {"comments": value}}}

    monkeypatch.setattr(collector, "github_api", api)
    threads = collector.review_threads("owner/repo", 2911)
    assert len(threads) == 2 and len(calls) == 6
    assert threads[1]["isResolved"]
    assert all(thread["comments"][-1]["body"] == "Expert correction" for thread in threads)


@pytest.mark.parametrize("database_id", [2**31 + 1, 2**53 + 1, 2**63 - 1, None])
@pytest.mark.parametrize("wire_type", [str, int])
def test_comment_ids_preserve_bigint_precision(collector, monkeypatch, database_id, wire_type):
    """Root and parent IDs remain exact JSON integers, including beyond float precision."""
    parent_id = database_id - 1 if database_id is not None else None

    def connection(query, variables, path):
        if "number" in variables:
            return [{"id": "thread"}]
        assert query.count("databaseId: fullDatabaseId") == 2
        return [
            {
                "databaseId": wire_type(database_id) if database_id is not None else None,
                "replyTo": {"databaseId": wire_type(parent_id) if parent_id is not None else None},
            }
        ]

    monkeypatch.setattr(collector, "connection", connection)
    threads = collector.review_threads("owner/repo", 2912)
    comment = json.loads(json.dumps(threads))[0]["comments"][0]
    assert comment["databaseId"] == database_id
    assert comment["replyTo"]["databaseId"] == parent_id
    if database_id is not None:
        assert type(comment["databaseId"]) is int
        assert type(comment["replyTo"]["databaseId"]) is int


@pytest.mark.parametrize("value", [True, 1.0, 0, -1, 2**63, str(2**63), "1.0", "1e3", "01", "", {}, []])
def test_malformed_comment_ids_fail_closed(collector, value):
    """Reject lossy, invalid, or out-of-range ID representations."""
    with pytest.raises(ValueError, match="comment ID"):
        collector.comment_database_id(value)


@pytest.mark.parametrize(
    "response",
    [
        {"errors": [{"message": "forbidden"}]},
        {"data": {"nodes": [], "pageInfo": {"hasNextPage": True, "endCursor": None}}},
        {"data": {"nodes": [], "pageInfo": {"hasNextPage": True, "endCursor": "unchanged"}}},
    ],
)
def test_partial_discussion_fails_closed(collector, monkeypatch, response):
    monkeypatch.setattr(collector, "github_api", lambda *args: response)
    with pytest.raises(ValueError):
        collector.connection("query", {}, ())


def test_snapshot_preserves_review_and_expert_input(collector, pr, monkeypatch):
    calls = []

    def api(endpoint, *, paginate=False):
        assert paginate
        calls.append(endpoint)
        return [{"id": 7, "body": "$(do not execute)\nExpert correction", "user": None, "state": "COMMENTED"}]

    monkeypatch.setattr(collector, "github_api", api)
    monkeypatch.setattr(
        collector, "review_threads", lambda *args: [{"isResolved": True, "comments": ["reply"]}]
    )
    snapshot = collector.discussion_snapshot("owner/repo", pr)
    assert snapshot["body"] == pr["body"]
    assert snapshot["comments"][0]["body"] == "$(do not execute)\nExpert correction"
    assert snapshot["reviews"][0]["state"] == "COMMENTED"
    assert snapshot["threads"][0]["comments"] == ["reply"]
    assert len(calls) == 2


def test_prepare_writes_safe_outputs(collector, pr, comment_event, monkeypatch, tmp_path):
    pr["base"]["ref"] = "branch-with-$(literal)"
    monkeypatch.setattr(collector, "review_target", lambda *args: pr)
    monkeypatch.setattr(
        collector, "discussion_snapshot", lambda *args: {"base_ref": pr["base"]["ref"], "body": "expert"}
    )
    destination, outputs = tmp_path / "discussion.json", tmp_path / "outputs"
    collector.prepare("issue_comment", comment_event, "owner/repo", destination, outputs)
    assert json.loads(destination.read_text())["base_ref"] == pr["base"]["ref"]
    assert outputs.read_text() == f"number=2911\nhead={'a' * 40}\nbase={'b' * 40}\nready=true\n"


def test_oversized_context_not_silently_truncated(collector, pr, comment_event, monkeypatch, tmp_path):
    monkeypatch.setattr(collector, "review_target", lambda *args: pr)
    monkeypatch.setattr(collector, "discussion_snapshot", lambda *args: {"body": "x" * 100})
    monkeypatch.setattr(collector, "MAX_CONTEXT_BYTES", 10)
    with pytest.raises(ValueError, match="incomplete"):
        collector.prepare(
            "issue_comment", comment_event, "owner/repo", tmp_path / "context", tmp_path / "outputs"
        )
    assert not (tmp_path / "outputs").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
