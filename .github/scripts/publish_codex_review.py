# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Publish untrusted review data using trusted, fail-closed approval policy."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

MARKER = "<!-- pypto-codex-review -->"
MAX_BYTES = 60000


def github_api(endpoint: str, payload: dict | None = None, *, paginate: bool = False) -> Any:
    """Call GitHub without interpolating model output into shell commands."""
    command = ["gh", "api", endpoint]
    if payload is not None:
        command += ["--method", "POST", "--input", "-"]
    if paginate:
        command += ["--paginate", "--slurp"]
    result = subprocess.run(
        command,
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return [item for page in data for item in page] if paginate else data


def unique_object(pairs: list[tuple[str, Any]]) -> dict:
    """Reject duplicate JSON fields instead of silently accepting the last one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate review field: {key}")
        result[key] = value
    return result


def load_review(path: Path) -> dict:
    """Validate every field before a review can authorize an approval."""
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Review must be a non-empty regular file of at most 60000 bytes")
    review = json.loads(path.read_text(), object_pairs_hook=unique_object)
    if not isinstance(review, dict) or set(review) != {"verdict", "summary", "findings"}:
        raise ValueError("Review must contain exactly verdict, summary, and findings")
    if review["verdict"] not in ("pass", "findings", "incomplete"):
        raise ValueError("Unknown review verdict")
    if not isinstance(review["summary"], str) or not review["summary"].strip():
        raise ValueError("Review summary must be a non-empty string")
    if not isinstance(review["findings"], list):
        raise ValueError("Review findings must be a list")
    for finding in review["findings"]:
        if not isinstance(finding, dict) or set(finding) != {"title", "body"}:
            raise ValueError("Each finding must contain exactly title and body")
        if any(not isinstance(value, str) or not value.strip() for value in finding.values()):
            raise ValueError("Finding title and body must be non-empty strings")
    if (review["verdict"] == "pass" and review["findings"]) or (
        review["verdict"] == "findings" and not review["findings"]
    ):
        raise ValueError("Review verdict contradicts findings")
    return review


def sensitive_path(path: str) -> bool:
    """Require human review for automation and agent-policy changes, including renames."""
    parts = PurePosixPath(path).parts
    return any(part in {".github", ".claude", ".codex", ".agents"} for part in parts) or (
        PurePosixPath(path).name in {"AGENTS.md", "CLAUDE.md", ".gitmodules"}
    )


def matches_revision(pr: dict, head: str, base: str) -> bool:
    """Check the PR still represents the complete revision pair that was reviewed."""
    return (
        pr["state"] == "open" and not pr["draft"] and pr["head"]["sha"] == head and pr["base"]["sha"] == base
    )


def approval_blocker(repo: str, endpoint: str, pr: dict, review: dict, enabled: bool) -> str | None:
    """Return why human review is needed, or None if all approval gates pass."""
    if not enabled:
        return "Automatic approval is disabled"
    if review["verdict"] != "pass":
        return "Review contains findings or is incomplete"
    files = github_api(f"{endpoint}/files?per_page=100", paginate=True)
    if len(files) != pr["changed_files"] or not files:
        return "Cannot verify the complete changed-file list"
    if any(
        sensitive_path(item[key])
        for item in files
        for key in ("filename", "previous_filename")
        if key in item
    ):
        return "Automation or agent-policy changes require human review"
    rules = github_api(f"repos/{repo}/rules/branches/{quote(pr['base']['ref'], safe='')}", paginate=True)
    if not any(
        rule["type"] == "pull_request" and rule["parameters"].get("dismiss_stale_reviews_on_push") is True
        for rule in rules
    ):
        return "Branch must dismiss stale approvals after new commits"
    return None


def publish(path: Path, repo: str, number: str, head: str, base: str, enabled: bool) -> str:
    """Post a review, approving only a complete clean review of the current revision."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid repository")
    if not re.fullmatch(r"[1-9][0-9]*", number):
        raise ValueError("Invalid pull request number")
    if any(not re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (head, base)):
        raise ValueError("Expected full commit SHA values")
    endpoint = f"repos/{repo}/pulls/{number}"
    pr = github_api(endpoint)
    if not matches_revision(pr, head, base):
        return "Skipped obsolete or non-reviewable PR"

    # Revoke our own prior approvals before parsing a replacement result. A bad
    # result or API failure must not leave an earlier automated pass in force.
    reviews = github_api(f"{endpoint}/reviews?per_page=100", paginate=True)
    for previous in reviews:
        if (
            previous["user"]["login"] == "github-actions[bot]"
            and previous["state"] == "APPROVED"
            and previous["body"].startswith(MARKER)
        ):
            command = [
                "gh",
                "api",
                f"{endpoint}/reviews/{previous['id']}/dismissals",
                "--method",
                "PUT",
                "--input",
                "-",
            ]
            subprocess.run(
                command,
                input=json.dumps({"message": "Superseded by a new Codex review"}),
                text=True,
                capture_output=True,
                check=True,
            )

    review = load_review(path)
    reason = approval_blocker(repo, endpoint, pr, review, enabled)
    approve = reason is None

    body = f"{MARKER}\n## Codex Review\n\nReviewed `{head}` against base `{base}`.\n\n"
    body += "> Automated assessment of untrusted PR content; not a guarantee of correctness.\n\n"
    body += review["summary"]
    for finding in review["findings"]:
        body += f"\n\n### {finding['title']}\n\n{finding['body']}"
    body += "\n\nApproval policy: passed." if approve else f"\n\nNo automatic approval: {reason}."
    if len(body.encode()) > MAX_BYTES:
        raise ValueError("Rendered review exceeds 60000 bytes")
    # Fetch immediately before posting, then again afterwards to close the race
    # with synchronize events. The review is always attached to the examined SHA.
    if not matches_revision(github_api(endpoint), head, base):
        return "Skipped PR updated during publication"
    posted = github_api(
        f"{endpoint}/reviews",
        {
            "commit_id": head,
            "event": "APPROVE" if approve else "COMMENT",
            "body": body,
        },
    )
    if approve and not matches_revision(github_api(endpoint), head, base):
        subprocess.run(
            ["gh", "api", f"{endpoint}/reviews/{posted['id']}/dismissals", "--method", "PUT", "--input", "-"],
            input=json.dumps({"message": "PR changed while Codex approval was being published"}),
            text=True,
            capture_output=True,
            check=True,
        )
        return "Dismissed approval because PR changed during publication"
    return "Approved reviewed commit" if approve else f"Commented: {reason}"


if __name__ == "__main__":
    print(
        publish(
            Path(sys.argv[1]),
            os.environ["GH_REPO"],
            os.environ["PR_NUMBER"],
            os.environ["REVIEW_HEAD_SHA"],
            os.environ["REVIEW_BASE_SHA"],
            os.environ.get("AUTO_APPROVE") == "true",
        )
    )
