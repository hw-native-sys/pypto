# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise review checkout preparation with real Git histories and missing blobs."""

import importlib
import json
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
    workflow = yaml.safe_load((root / ".github/workflows/ai-review.yml").read_text())
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


def test_gemini_workflow_resumes_only_a_validated_pr_session(tmp_path, review_history, monkeypatch):
    """Use real Git/launcher arguments and native envelopes; substitute only GitHub and the index volume."""
    origin, base, head, merge_base, _ = review_history
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / ".github/scripts"))
    driver = importlib.import_module("run_gemini_review")
    job = yaml.safe_load((root / ".github/workflows/ai-review.yml").read_text())["jobs"]["gemini"]
    steps = {step.get("id"): step for step in job["steps"]}
    prepare = steps["workspace"]["run"]
    remove = job["steps"][-1]["run"]
    assert "run_gemini_review.py review --model gemini-4-neon-high --effort high" in steps["model"]["run"]
    remote = tmp_path / "owner/repo.git"
    remote.parent.mkdir()
    git(tmp_path, "clone", "--bare", str(origin), str(remote))
    for name, value in {
        "GITHUB_SERVER_URL": tmp_path.as_uri(),
        "GITHUB_REPOSITORY": "owner/repo",
        "GH_REPO": "owner/repo",
        "REPOSITORY_ID": "42",
        "GITHUB_REPOSITORY_ID": "42",
        "GITHUB_RUN_ID": "111",
        "GITHUB_RUN_ATTEMPT": "2",
        "PR_NUMBER": "12",
        "RUNNER_TEMP": str(tmp_path),
        "BASE_SHA": base,
        "HEAD_SHA": head,
        "HOME": str(tmp_path / "home"),
        "GH_TOKEN": "host-only",
        "GITHUB_TOKEN": "host-only",
        "ACTIONS_RUNTIME_TOKEN": "host-only",
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }.items():
        monkeypatch.setenv(name, value)
    snapshot = tmp_path / "ai-review-context/discussion.json"
    snapshot.parent.mkdir()
    snapshot.write_text(
        json.dumps(
            {
                "repository": "owner/repo",
                "number": 12,
                "base": base,
                "head": head,
                "base_ref": "main",
                "body": "review context",
            }
        )
    )
    session_id = "12345678-1234-1234-1234-123456789abc"
    record = tmp_path / "native.json"
    args_file = tmp_path / "arguments"
    valid = {"verdict": "pass", "summary": "Examined all changes.", "findings": []}
    native = {"status": "SUCCESS", "conversation_id": session_id, "structured_output": valid}
    record.write_text(json.dumps(native))
    launcher = Path.home() / ".local/share/pypto-gemini-review/run"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        "#!/bin/bash\nset -eu\n"
        'test -z "${GH_TOKEN-}${GITHUB_TOKEN-}${ACTIONS_RUNTIME_TOKEN-}${GITHUB_OUTPUT-}"\n'
        f'printf "%s\\n" "$@" >"{args_file}"\n'
        f'cp "{record}" "${{@: -1}}/result.json"\n'
    )
    launcher.chmod(0o755)
    saved = {}

    def checkpoint(name, session=None):
        assert name == "pypto-gemini-sessions-42-12-index"
        if session is not None:
            saved[name] = session
        return saved.get(name, "")

    def authorize(repo, number, repo_id, mode, sha, *, provider):
        assert (repo, number, repo_id, mode, sha, provider) == (
            "owner/repo",
            "12",
            "42",
            "review",
            head,
            "gemini",
        )
        return "ready"

    monkeypatch.setattr(driver, "reconcile", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "checkpoint", checkpoint)
    monkeypatch.setattr(driver, "manage_session", authorize)
    for turn in ("new", "resume", "wrong_id", "failed_result"):
        subprocess.run(["bash", "-c", prepare], check=True, capture_output=True)
        workdir = Path((tmp_path / "outputs").read_text().splitlines()[-1].removeprefix("path="))
        inputs = workdir / "input"
        assert (inputs / "pr.diff").read_text().strip() == git(origin, "diff", "--binary", merge_base, head)
        assert git(inputs / "workspace", "rev-parse", "HEAD") == head
        assert (inputs / "workspace/python/pypto/ir/compile.py").is_file()
        if turn == "wrong_id":
            record.write_text(
                json.dumps(dict(native, conversation_id="aaaaaaaa-1234-1234-1234-123456789abc"))
            )
            with pytest.raises(ValueError, match="requested conversation"):
                driver.review(workdir, "gemini-4-neon-high", "high")
            assert not (workdir / "response.txt").exists()
        elif turn == "failed_result":
            record.write_text(json.dumps(dict(native, status="ERROR")))
            with pytest.raises(ValueError, match="did not complete"):
                driver.review(workdir, "gemini-4-neon-high", "high")
            assert not (workdir / "response.txt").exists()
        else:
            assert driver.review(workdir, "gemini-4-neon-high", "high")
            assert json.loads((workdir / "response.txt").read_text()) == valid
        args = args_file.read_text().splitlines()
        assert args[:7] == [
            "review-session",
            "--model",
            "gemini-4-neon-high",
            "--effort",
            "high",
            "--session-volume",
            "pypto-gemini-sessions-42-12",
        ]
        assert args[7:-2] == ([] if turn == "new" else ["--conversation", session_id])
        assert saved == {"pypto-gemini-sessions-42-12-index": session_id}
        assert (inputs / "discussion.json").read_bytes() == snapshot.read_bytes()
        assert (inputs / "schema.json").read_bytes() == (root / ".github/ai-review-schema.json").read_bytes()
        assert "review-discussion.json" in (inputs / "prompt.txt").read_text()
        subprocess.run(["bash", "-c", remove], check=True, env=os.environ | {"REVIEW_DIR": str(workdir)})
        assert not workdir.exists()


def test_gemini_auth_maintenance_runs_without_pr_data(tmp_path):
    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/ai-review-auth-maintenance.yml").read_text())
    step = workflow["jobs"]["gemini-auth-maintenance"]["steps"][-1]
    launcher = tmp_path / ".local/share/pypto-gemini-review/run"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        "#!/bin/sh\nset -eu\n"
        'test "$#" = 3 && test "$1" = auth-maintenance && test "$2" = --model\n'
        'test -z "${GH_TOKEN-}${GITHUB_TOKEN-}${ACTIONS_RUNTIME_TOKEN-}" && test -z "$(ls -A)"\n'
        f'pwd > "{tmp_path}/refreshed"\n'
    )
    launcher.chmod(0o755)
    subprocess.run(
        ["bash", "-c", step["run"]],
        check=True,
        cwd=root,
        env=os.environ
        | {
            "HOME": str(tmp_path),
            "RUNNER_TEMP": str(tmp_path),
            "GH_TOKEN": "host-only",
            "GITHUB_TOKEN": "host-only",
            "ACTIONS_RUNTIME_TOKEN": "host-only",
        },
    )
    empty = Path((tmp_path / "refreshed").read_text().strip())
    assert empty.parent == tmp_path and not empty.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
