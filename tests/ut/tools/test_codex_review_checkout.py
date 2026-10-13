# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise review checkout preparation with real Git histories and missing blobs."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml


def git(repository: Path, *arguments: str) -> str:
    """Run Git against a test repository and return its output."""
    return subprocess.run(
        ["git", "-C", str(repository), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def review_history(tmp_path, monkeypatch):
    """Create diverged base/head commits with an older compile.py blob."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Review checkout test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "review-checkout@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Review checkout test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "review-checkout@example.invalid")
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "--initial-branch=main")
    source = origin / "python/pypto/ir/compile.py"
    source.parent.mkdir(parents=True)
    source.write_text("# original compile implementation\n")
    git(origin, "add", "python/pypto/ir/compile.py")
    git(origin, "commit", "-m", "Create merge base")
    merge_base = git(origin, "rev-parse", "HEAD")
    old_blob = git(origin, "rev-parse", "HEAD:python/pypto/ir/compile.py")
    source.write_text("# updated compile implementation\n")
    git(origin, "commit", "-am", "Update compile implementation")
    head = git(origin, "rev-parse", "HEAD")
    git(origin, "checkout", "-b", "base", merge_base)
    (origin / "base-only.txt").write_text("Base advanced independently\n")
    git(origin, "add", "base-only.txt")
    git(origin, "commit", "-m", "Advance base")
    base = git(origin, "rev-parse", "HEAD")
    return origin, base, head, merge_base, old_blob


@pytest.fixture
def review_steps():
    """Load the actual trusted shell steps instead of reimplementing preparation."""
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/codex-review.yml").read_text())
    return {step["name"]: step for step in workflow["jobs"]["review"]["steps"]}


def verify_checkout(repository, steps, base, head):
    """Execute the workflow's preparation step with the test routing identity."""
    step = steps["Fetch and verify complete review objects"]
    return subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=repository,
        env=os.environ
        | {"BASE_SHA": base, "HEAD_SHA": head, "TRUSTED_WORKFLOW_SHA": base}
        | {"GIT_NO_LAZY_FETCH": step["env"]["GIT_NO_LAZY_FETCH"]},
        capture_output=True,
        text=True,
        check=False,
    )


def damaged_checkout(destination, history):
    """Keep current source and commits while removing the old file-content object."""
    origin, _, head, _, old_blob = history
    git(destination.parent, "clone", "--no-hardlinks", str(origin), str(destination))
    git(destination, "checkout", "--detach", head)
    (destination / ".git/objects" / old_blob[:2] / old_blob[2:]).unlink()
    assert (destination / "python/pypto/ir/compile.py").read_text() == "# updated compile implementation\n"
    assert git(destination, "rev-parse", "HEAD") == head


def test_fresh_checkout_ignores_incomplete_runner_repository(tmp_path, review_history, review_steps):
    """A fresh checkout recovers readable merge-base contents despite a broken cache."""
    origin, base, head, merge_base, old_blob = review_history
    workspace = tmp_path / "runner-workspace"
    damaged_checkout(workspace, review_history)
    git(workspace, "config", "remote.origin.promisor", "true")
    git(workspace, "config", "remote.origin.partialclonefilter", "blob:none")
    output = tmp_path / "step-output"
    subprocess.run(
        ["bash", "-c", review_steps["Create a fresh review checkout directory"]["run"]],
        cwd=workspace,
        env=os.environ | {"GITHUB_WORKSPACE": str(workspace), "GITHUB_OUTPUT": str(output)},
        check=True,
        capture_output=True,
        text=True,
    )
    relative = output.read_text().strip().removeprefix("path=")
    fresh = workspace / relative
    assert fresh.is_dir() and not (fresh / ".git").exists()
    assert fresh.stat().st_mode & 0o777 == 0o755
    git(workspace, "clone", "--no-local", "--no-checkout", str(origin), str(fresh))
    git(fresh, "checkout", "--detach", head)
    result = verify_checkout(fresh, review_steps, base, head)
    assert result.returncode == 0, result.stderr
    assert f"Complete review diff verified: {merge_base}..{head}" in result.stdout
    assert git(fresh, "cat-file", "-p", old_blob) == "# original compile implementation"


def test_missing_historical_blob_stops_review(tmp_path, review_history, review_steps):
    """Commit presence and current source cannot pass the complete-object gate."""
    _, base, head, _, _ = review_history
    checkout = tmp_path / "incomplete-checkout"
    damaged_checkout(checkout, review_history)
    result = verify_checkout(checkout, review_steps, base, head)
    assert result.returncode != 0
    assert "Complete review diff verified" not in result.stdout


def test_partial_clone_stops_review(tmp_path, review_history, review_steps):
    """Reject partial-clone metadata even when the immediate diff is readable."""
    origin, base, head, _, _ = review_history
    checkout = tmp_path / "partial-checkout"
    git(tmp_path, "clone", "--no-local", str(origin), str(checkout))
    git(checkout, "checkout", "--detach", head)
    git(checkout, "config", "remote.origin.promisor", "true")
    git(checkout, "config", "remote.origin.partialclonefilter", "blob:none")
    result = verify_checkout(checkout, review_steps, base, head)
    assert result.returncode != 0
    assert "Review checkout must not be a partial clone" in result.stderr


def test_preparation_and_container_use_fresh_checkout(review_steps):
    """The validated repository must be the one mounted read-only for review."""
    checkout = review_steps["Checkout PR head"]["with"]
    verification = review_steps["Fetch and verify complete review objects"]
    container = review_steps["Review in an isolated container"]
    assert checkout["path"] == verification["working-directory"] == container["working-directory"]
    assert checkout["fetch-depth"] == 0 and checkout["filter"] == ""
    assert checkout["persist-credentials"] is False
    names = list(review_steps)
    assert names.index("Fetch and verify complete review objects") < names.index(
        "Review in an isolated container"
    )
    assert "src=$PWD,dst=/workspace,readonly" in container["run"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
