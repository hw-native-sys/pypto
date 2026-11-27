# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the host-managed Gemini container with a PR-scoped, verified conversation checkpoint."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from cleanup_ai_review_resources import reconcile
from manage_ai_review_session import manage_session
from publish_ai_review import GEMINI_MARKER, MAX_BYTES, parse_review, unique_object

UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
POLICY = Path(__file__).resolve().parents[1]


def launch(arguments: list[str], *, cwd: str | None = None) -> None:
    """Keep Actions credentials and runner command files out of the host adapter."""
    allowed = (
        "HOME",
        "PATH",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "RUNNER_TEMP",
        "XDG_RUNTIME_DIR",
    )
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    if arguments[0] == "review-session":
        # Resource ownership labels only, never the Actions command files or credentials.
        for key in ("GITHUB_REPOSITORY_ID", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
            env[key] = os.environ[key]
    subprocess.run(
        [str(Path.home() / ".local/share/pypto-gemini-review/run"), *arguments],
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def checkpoint(index: str, session_id: str | None = None) -> str:
    """Only a trusted, offline helper can read or atomically replace this index."""
    script = "if test -e /index/id; then cat /index/id; fi"
    mount = f"type=volume,src={index},dst=/index,volume-nocopy"
    args = []
    if session_id is None:
        mount += ",readonly"
    else:
        if not UUID.fullmatch(session_id):
            raise ValueError("Invalid Gemini conversation ID")
        script = 'umask 077; printf "%s\\n" "$1" > /index/id.tmp && mv /index/id.tmp /index/id'
        args = ["sh", session_id]
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "0:0",
            "--mount",
            mount,
            "--entrypoint",
            "/bin/sh",
            os.environ["GEMINI_REVIEW_IMAGE"],
            "-ec",
            script,
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    saved = result.stdout.rstrip("\n")
    if saved and not UUID.fullmatch(saved):
        raise ValueError("Invalid Gemini conversation checkpoint")
    return saved


def prepare_context(inputs: Path, discussion: Path, repo: str, number: str, base: str, head: str) -> None:
    """Bind the discussion and the workflow's trusted review policy to the fetched revision."""
    snapshot = json.loads(discussion.read_bytes(), object_pairs_hook=unique_object)
    if (
        snapshot["repository"],
        snapshot["number"],
        snapshot["base"],
        snapshot["head"],
        snapshot["base_ref"],
    ) != (repo, int(number), base, head, "main"):
        raise ValueError("Gemini discussion does not match the authorized PR revision")
    shutil.copyfile(discussion, inputs / "discussion.json")
    shutil.copyfile(POLICY / "ai-review-schema.json", inputs / "schema.json")
    policy = (POLICY / "ai-review-prompt.txt").read_text()
    (inputs / "prompt.txt").write_text(
        f"You are the Gemini reviewer ({GEMINI_MARKER}). Base {base}, head {head}. "
        "The complete diff is in /pr.diff and the read-only Git checkout is in /workspace.\n" + policy
    )
    for name in ("discussion.json", "schema.json", "prompt.txt"):
        (inputs / name).chmod(0o644)


def extract_result(path: Path, previous: str) -> tuple[str, str]:
    """Accept only the successful native result for the requested conversation."""
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 1024 * 1024:
        raise ValueError("Invalid Gemini result file")
    result = json.loads(path.read_bytes(), object_pairs_hook=unique_object)
    if not isinstance(result, dict) or result.get("status") != "SUCCESS":
        raise ValueError("Gemini did not complete its review")
    session_id = result.get("conversation_id")
    if (
        not isinstance(session_id, str)
        or not UUID.fullmatch(session_id)
        or (previous and session_id != previous)
    ):
        raise ValueError("Gemini did not return the requested conversation")
    text = json.dumps(result.get("structured_output"), ensure_ascii=False)
    if len(text.encode()) > MAX_BYTES:
        raise ValueError("Gemini review exceeds the publication limit")
    parse_review(text)
    return session_id, text


def review(workdir: Path, model: str, effort: str) -> bool:
    """Resume only this PR and record its ID after a complete schema-valid CLI turn."""
    repo, number, repo_id = (os.environ[k] for k in ("GH_REPO", "PR_NUMBER", "REPOSITORY_ID"))
    base, head = os.environ["BASE_SHA"], os.environ["HEAD_SHA"]
    reconcile(repo, repo_id, provider="gemini")
    if manage_session(repo, number, repo_id, "review", head, provider="gemini") == "skipped":
        return False
    volume = f"pypto-gemini-sessions-{repo_id}-{number}"
    previous = checkpoint(volume + "-index")
    inputs = workdir / "input"
    prepare_context(
        inputs,
        Path(os.environ["RUNNER_TEMP"]) / "ai-review-context/discussion.json",
        repo,
        number,
        base,
        head,
    )
    arguments = ["review-session", "--model", model, "--effort", effort, "--session-volume", volume]
    if previous:
        arguments += ["--conversation", previous]
    for attempt in range(1, 4):
        results = workdir / f"attempt-{attempt}"
        results.mkdir(mode=0o700)
        try:
            launch([*arguments, str(inputs), str(results)])
        except subprocess.CalledProcessError as error:
            print(f"Gemini launcher attempt {attempt} exited with status {error.returncode}", file=sys.stderr)
            if attempt == 3:
                raise
            time.sleep(10)
        else:
            break
    session_id, text = extract_result(results / "result.json", previous)
    (workdir / "response.txt").write_text(text, encoding="utf-8")
    checkpoint(volume + "-index", session_id)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    session = commands.add_parser("review")
    session.add_argument("--model", required=True)
    session.add_argument("--effort", required=True)
    session.add_argument("workdir", type=Path)
    commands.add_parser("cleanup")
    commands.add_parser("auth-maintenance").add_argument("--model", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    state = Path.home() / ".local/state/pypto-gemini-review"
    # Closing/reopening a PR and a review must observe the same volume generation.
    # The host launcher uses its own OAuth lock for login and credential refresh.
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(state / "ci.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.command == "cleanup":
            reconcile(os.environ["GH_REPO"], os.environ["REPOSITORY_ID"], provider="gemini")
            print(
                manage_session(
                    os.environ["GH_REPO"],
                    os.environ["PR_NUMBER"],
                    os.environ["REPOSITORY_ID"],
                    "cleanup",
                    provider="gemini",
                )
            )
        elif args.command == "auth-maintenance":
            with TemporaryDirectory(dir=os.environ.get("RUNNER_TEMP"), prefix="gemini-auth-") as empty:
                launch(["auth-maintenance", "--model", args.model], cwd=empty)
        elif review(args.workdir, args.model, args.effort):
            with Path(os.environ["GITHUB_OUTPUT"]).open("a") as outputs:
                outputs.write(f"artifact={args.workdir / 'response.txt'}\ncompleted=true\n")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f"Gemini host command failed with status {error.returncode}", file=sys.stderr)
        sys.exit(error.returncode if error.returncode > 0 else 1)
    except (OSError, KeyError, TypeError, ValueError, RecursionError):
        sys.exit("Gemini session, review result, or credential maintenance could not be verified")
