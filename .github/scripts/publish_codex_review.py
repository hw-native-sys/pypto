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
DIFF_READ_ERRORS = (OSError, subprocess.CalledProcessError, KeyError, TypeError, ValueError)


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
        # Accept older artifacts on workflow reruns; new output includes location.
        if not isinstance(finding, dict) or set(finding) not in (
            {"title", "body"},
            {"title", "body", "location"},
            {"title", "body", "location", "existing_comment_id"},
        ):
            raise ValueError("Each finding must contain title, body, and optional location")
        reference = finding.get("existing_comment_id")
        if reference is not None and (type(reference) is not int or reference <= 0):
            raise ValueError("Existing comment ID must be a positive integer or null")
        if any(not isinstance(finding[key], str) or not finding[key].strip() for key in ("title", "body")):
            raise ValueError("Finding title and body must be non-empty strings")
    if (review["verdict"] == "pass" and review["findings"]) or (
        review["verdict"] == "findings" and not review["findings"]
    ):
        raise ValueError("Review verdict contradicts findings")
    return review


def extract_session(events: Path, output: Path) -> str:
    """Validate a completed exec event stream and return its exact root session ID."""
    session_id = None
    final_text = None
    completed = False
    with events.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("type") == "thread.started":
                candidate = event.get("thread_id", "")
                if session_id is not None or not re.fullmatch(
                    r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", candidate
                ):
                    raise ValueError("Invalid or ambiguous Codex session ID")
                session_id = candidate
            elif event.get("type") == "item.completed":
                item = event["item"]
                if item.get("type") == "agent_message":
                    final_text = item.get("text")
            elif event.get("type") == "turn.completed":
                completed = True
            elif event.get("type") in {"turn.failed", "error"}:
                raise ValueError("Codex session did not complete successfully")
    if not completed or session_id is None or not isinstance(final_text, str):
        raise ValueError("Missing completed Codex review or session ID")
    output.write_text(final_text)
    load_review(output)
    return session_id


def finding_location(finding: dict) -> dict | None:
    """Treat invalid model locations as unanchored findings, never discard their text."""
    location = finding.get("location")
    if not isinstance(location, dict) or set(location) != {"path", "line", "side"}:
        return None
    path = location["path"]
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or any(ord(char) < 32 for char in path)
        or type(location["line"]) is not int
        or location["line"] <= 0
        or location["side"] not in ("LEFT", "RIGHT")
    ):
        return None
    return location


def patch_lines(patch: str) -> set[tuple[str, int]]:
    """Extract side-specific anchors, rejecting incomplete or malformed hunks."""
    anchors: set[tuple[str, int]] = set()
    old = new = old_left = new_left = 0
    active = False
    for text in patch.splitlines():
        header = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*", text)
        if header:
            if old_left or new_left:
                return set()
            old_start, old_count, new_start, new_count = header.groups()
            old, new = int(old_start), int(new_start)
            old_left = int(old_count) if old_count is not None else 1
            new_left = int(new_count) if new_count is not None else 1
            active = True
            continue
        if text == "\\ No newline at end of file":
            continue
        if not active or not text or text[0] not in " +-":
            return set()
        if text[0] in " -":
            if old_left <= 0:
                return set()
            # GitHub uses RIGHT for context, LEFT for deleted lines.
            if text[0] == "-":
                anchors.add(("LEFT", old))
            old += 1
            old_left -= 1
        if text[0] in " +":
            if new_left <= 0:
                return set()
            anchors.add(("RIGHT", new))
            new += 1
            new_left -= 1
    return set() if old_left or new_left else anchors


def summary_finding(finding: dict) -> str:
    """Preserve reported coordinates even when no verified anchor or link is available."""
    text = f"### {finding['title']}\n\n{finding['body']}"
    location = finding_location(finding)
    if location:
        text += f"\n\nReported location: {location['path']}:{location['line']} ({location['side']})."
    return text


def render_findings(
    repo: str, endpoint: str, head: str, merge_base: str | None, findings: list
) -> tuple[str, list]:
    """Split findings into validated inline comments and a lossless summary fallback."""
    if not findings:
        return "", []
    summary_only = "".join(f"\n\n{summary_finding(finding)}" for finding in findings)
    if merge_base is None or not any(
        finding_location(finding) or finding.get("existing_comment_id") for finding in findings
    ):
        return summary_only, []
    try:
        files = github_api(f"{endpoint}/files?per_page=100", paginate=True)
        previous = github_api(f"{endpoint}/comments?per_page=100", paginate=True)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        # Location and deduplication metadata are optional; keep findings visible.
        return summary_only, []
    existing = {
        item["id"]: item
        for item in previous
        if item.get("id")
        and not item.get("in_reply_to_id")
        and (item.get("user") or {}).get("login") == "github-actions[bot]"
        and (item.get("body") or "").startswith(MARKER)
    }
    by_path = {item["filename"]: item for item in files}
    anchors = {path: patch_lines(item.get("patch", "")) for path, item in by_path.items()}
    seen = {
        (item["path"], item.get("line"), item.get("side"), item["body"])
        for item in previous
        if (item.get("user") or {}).get("login") == "github-actions[bot]"
        and item.get("original_commit_id") == head
        and (item.get("body") or "").startswith(MARKER)
    }
    summary = ""
    comments = []
    for finding in findings:
        text = f"### {finding['title']}\n\n{finding['body']}"
        reference = finding.get("existing_comment_id")
        if reference in existing:
            url = f"https://github.com/{repo}/pull/{endpoint.rsplit('/', 1)[-1]}#discussion_r{reference}"
            summary += f"\n\n{summary_finding(finding)}"
            summary += f"\n\n[Existing review thread]({url}); no duplicate inline comment posted."
            continue
        location = finding_location(finding)
        if location:
            path, line, side = location["path"], location["line"], location["side"]
            body = f"{MARKER}\n{text}"
            key = (path, line, side, body)
            if (side, line) in anchors.get(path, set()):
                if key in seen:
                    summary += f"\n\n{summary_finding(finding)}"
                    summary += "\n\nAn identical inline comment already exists for this commit."
                    continue
                # Keep oversized batches in the summary rather than lose findings.
                if len(comments) < 100:
                    comments.append({"path": path, "line": line, "side": side, "body": body})
                    seen.add(key)
                    continue
            revision = head
            if side == "LEFT":
                revision = merge_base
                path = by_path.get(path, {}).get("previous_filename", path)
            url = f"https://github.com/{repo}/blob/{revision}/{quote(path, safe='/')}#L{line}"
            text += f"\n\n[Reported code location]({url}) (not published inline)."
        summary += f"\n\n{text}"
    if comments:
        summary = f"\n\n{len(comments)} finding(s) posted as inline comments." + summary
    return summary, comments


def sensitive_path(path: str) -> bool:
    """Require human review for automation and agent-policy changes, including renames."""
    parts = PurePosixPath(path).parts
    return any(part in {".github", ".claude", ".codex", ".agents"} for part in parts) or (
        PurePosixPath(path).name in {"AGENTS.md", "AGENTS.override.md", "CLAUDE.md", ".gitmodules"}
    )


def merge_base_sha(repo: str, base: str, head: str) -> str:
    """Read the merge base used to construct the reviewed PR diff."""
    if any(not re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (base, head)):
        raise ValueError("Expected full commit SHA values")
    comparison = github_api(f"repos/{repo}/compare/{base}...{head}")
    candidate = comparison["merge_base_commit"]["sha"]
    if not isinstance(candidate, str) or not re.fullmatch(r"[0-9a-f]{40}", candidate):
        raise ValueError("GitHub compare response has an invalid merge base")
    return candidate


def optional_merge_base_sha(repo: str, base: str, head: str) -> str | None:
    """Fail closed on an unavailable diff check without hiding review findings."""
    try:
        return merge_base_sha(repo, base, head)
    except DIFF_READ_ERRORS:
        return None


def matches_revision(pr: dict, head: str, base_ref: str) -> bool:
    """Check the PR still has the reviewed head and target branch."""
    return (
        pr["state"] == "open"
        and not pr["draft"]
        and pr["head"]["sha"] == head
        and pr["base"]["ref"] == base_ref
    )


def approval_blocker(endpoint: str, pr: dict, review: dict, enabled: bool) -> str | None:
    """Return why human review is needed, or None if all approval gates pass."""
    if not enabled:
        return "Automatic approval is disabled"
    if pr["user"]["login"] == "github-actions[bot]":
        return "GitHub Actions cannot approve its own pull request"
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
    return None


def pull_request_endpoint(repo: str, number: str) -> str:
    """Validate the target before constructing a pull-request API endpoint."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid repository")
    if not re.fullmatch(r"[1-9][0-9]*", number):
        raise ValueError("Invalid pull request number")
    return f"repos/{repo}/pulls/{number}"


def revoke_approvals(repo: str, number: str) -> None:
    """Dismiss this workflow's approvals without relying on review eligibility."""
    endpoint = pull_request_endpoint(repo, number)
    reviews = github_api(f"{endpoint}/reviews?per_page=100", paginate=True)
    for previous in reviews:
        if (
            previous["user"]["login"] == "github-actions[bot]"
            and previous["state"] == "APPROVED"
            and (previous["body"] or "").startswith(MARKER)
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


def publish(path: Path, repo: str, number: str, head: str, base: str, base_ref: str, enabled: bool) -> str:
    """Post a review, approving only a complete clean review of the current revision."""
    endpoint = pull_request_endpoint(repo, number)
    if any(not re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (head, base)):
        raise ValueError("Expected full commit SHA values")
    pr = github_api(endpoint)
    if not matches_revision(pr, head, base_ref):
        return "Skipped obsolete or non-reviewable PR"

    # A bad replacement result must not leave our earlier approval in force.
    revoke_approvals(repo, number)
    review = load_review(path)
    reviewed_merge_base = optional_merge_base_sha(repo, base, head)
    current_merge_base = optional_merge_base_sha(repo, pr["base"]["sha"], head)
    diff_verified = reviewed_merge_base is not None and current_merge_base == reviewed_merge_base
    reason = approval_blocker(endpoint, pr, review, enabled)
    if not diff_verified:
        reason = "Cannot verify that the reviewed PR diff is current"
    approve = reason is None

    body = f"{MARKER}\n## Codex Review\n\nReviewed `{head}` against base `{base}`.\n\n"
    body += "> Automated assessment of untrusted PR content; not a guarantee of correctness.\n\n"
    body += review["summary"]
    findings_body, comments = render_findings(
        repo, endpoint, head, reviewed_merge_base if diff_verified else None, review["findings"]
    )
    # Fetch immediately before posting, then again afterwards to close the race
    # with synchronize events. The review is always attached to the examined SHA.
    latest = github_api(endpoint)
    if not matches_revision(latest, head, base_ref):
        return "Skipped PR updated during publication"
    if diff_verified and optional_merge_base_sha(repo, latest["base"]["sha"], head) != reviewed_merge_base:
        diff_verified = False
        reason = "Cannot verify that the reviewed PR diff is current"
        approve = False
        findings_body, comments = render_findings(repo, endpoint, head, None, review["findings"])
    body += findings_body
    body += "\n\nApproval policy: passed." if approve else f"\n\nNo automatic approval: {reason}."
    if len(body.encode()) + sum(len(item["body"].encode()) for item in comments) > MAX_BYTES:
        raise ValueError("Rendered review exceeds 60000 bytes")
    posted = github_api(
        f"{endpoint}/reviews",
        {
            "commit_id": head,
            "event": "APPROVE" if approve else "COMMENT",
            "body": body,
            **({"comments": comments} if comments else {}),
        },
    )
    if approve:
        try:
            latest = github_api(endpoint)
            still_current = matches_revision(latest, head, base_ref) and (
                optional_merge_base_sha(repo, latest["base"]["sha"], head) == reviewed_merge_base
            )
        except DIFF_READ_ERRORS:
            still_current = False
        if not still_current:
            subprocess.run(
                [
                    "gh",
                    "api",
                    f"{endpoint}/reviews/{posted['id']}/dismissals",
                    "--method",
                    "PUT",
                    "--input",
                    "-",
                ],
                input=json.dumps({"message": "PR changed or diff verification failed during publication"}),
                text=True,
                capture_output=True,
                check=True,
            )
            return "Dismissed approval because PR changed or diff verification failed"
    return "Approved reviewed commit" if approve else f"Commented: {reason}"


if __name__ == "__main__":
    if sys.argv[1] == "--extract-session":
        print(extract_session(Path(sys.argv[2]), Path(sys.argv[3])))
    elif sys.argv[1] == "--revoke":
        revoke_approvals(os.environ["GH_REPO"], os.environ["PR_NUMBER"])
    else:
        print(
            publish(
                Path(sys.argv[1]),
                os.environ["GH_REPO"],
                os.environ["PR_NUMBER"],
                os.environ["REVIEW_HEAD_SHA"],
                os.environ["REVIEW_BASE_SHA"],
                os.environ["REVIEW_BASE_REF"],
                os.environ.get("AUTO_APPROVE") == "true",
            )
        )
