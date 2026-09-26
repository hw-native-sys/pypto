# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Collect GitHub discussion as untrusted review evidence, using trusted code only."""

import json
import os
import re
import sys
from pathlib import Path

from publish_codex_review import github_api, pull_request_endpoint

COMMAND = "@pypto-codex review"
MAX_CONTEXT_BYTES = 8 * 1024 * 1024


def requests_review(body: str) -> bool:
    """Accept an exact standalone command outside Markdown fences and quotes."""
    fence = None
    for line in body.splitlines():
        stripped = line.strip()
        match = re.match(r"^(`{3,}|~{3,})", stripped)
        if match:
            marker = match[1]
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is None and stripped.casefold() == COMMAND:
            return True
    return False


def review_target(event_name: str, event: dict, repo: str) -> dict | None:
    """Resolve a current PR and authorize comment commands before starting a runner."""
    if event_name == "pull_request_target":
        candidate = event["pull_request"]
        if event["action"] == "edited" and not event.get("changes", {}).get("base"):
            return None
        if event["sender"]["type"] == "Bot" and event["action"] != "edited":
            return None
    elif event_name == "issue_comment":
        comment = event["comment"]
        if (
            event["action"] != "created"
            or comment["user"]["type"] != "User"
            or not requests_review(comment.get("body") or "")
        ):
            return None
        candidate = event.get("pull_request") or event.get("issue", {})
        if event_name == "issue_comment" and not candidate.get("pull_request"):
            return None
    else:
        return None
    number = str(candidate["number"])
    pr = github_api(pull_request_endpoint(repo, number))
    if pr["state"] != "open" or pr["draft"] or pr["base"]["repo"]["full_name"] != repo:
        return None
    if event_name == "pull_request_target" and (
        candidate["head"]["sha"] != pr["head"]["sha"] or candidate["base"]["ref"] != pr["base"]["ref"]
    ):
        return None
    if event_name != "pull_request_target":
        author = event["comment"]["user"]["login"]
        if author != pr["user"]["login"]:
            if not re.fullmatch(r"[A-Za-z0-9-]+", author):
                raise ValueError("Invalid GitHub comment author")
            permission = github_api(f"repos/{repo}/collaborators/{author}/permission")["permission"]
            if permission not in {"write", "maintain", "admin"}:
                return None
    if any(not re.fullmatch(r"[0-9a-f]{40}", pr[side]["sha"]) for side in ("head", "base")):
        raise ValueError("Expected full PR commit SHA values")
    return pr


def connection(query: str, variables: dict, path: tuple[str, ...]) -> list[dict]:
    """Paginate one GraphQL connection, rejecting errors and missing cursors."""
    nodes = []
    cursor = None
    cursors = set()
    while True:
        response = github_api("graphql", {"query": query, "variables": {**variables, "cursor": cursor}})
        if response.get("errors"):
            raise ValueError("GitHub discussion query failed")
        value = response["data"]
        for key in path:
            value = value[key]
        nodes.extend(value["nodes"])
        page = value["pageInfo"]
        if not page["hasNextPage"]:
            return nodes
        cursor = page["endCursor"]
        if not cursor or cursor in cursors:
            raise ValueError("GitHub discussion pagination did not advance")
        cursors.add(cursor)


def review_threads(repo: str, number: int) -> list[dict]:
    """Fetch all threads, including every reply and resolved/outdated state."""
    owner, name = repo.split("/")
    threads = connection(
        """query($owner:String!, $name:String!, $number:Int!, $cursor:String) {
          repository(owner:$owner, name:$name) { pullRequest(number:$number) {
            reviewThreads(first:100, after:$cursor) {
              nodes { id isResolved isOutdated path line originalLine diffSide
                resolvedBy { login } }
              pageInfo { hasNextPage endCursor }
            }
          } }
        }""",
        {"owner": owner, "name": name, "number": number},
        ("repository", "pullRequest", "reviewThreads"),
    )
    # Separate pagination prevents silently losing replies after the first 100.
    for thread in threads:
        thread["comments"] = connection(
            """query($id:ID!, $cursor:String) {
              node(id:$id) { ... on PullRequestReviewThread {
                comments(first:100, after:$cursor) {
                  nodes { databaseId body url createdAt updatedAt author { login }
                    replyTo { databaseId } commit { oid } }
                  pageInfo { hasNextPage endCursor }
                }
              } }
            }""",
            {"id": thread["id"]},
            ("node", "comments"),
        )
    return threads


def discussion_snapshot(repo: str, pr: dict) -> dict:
    """Include descriptions, review bodies, conversation comments and complete threads."""
    number = str(pr["number"])
    endpoint = pull_request_endpoint(repo, number)
    comments = github_api(f"repos/{repo}/issues/{number}/comments?per_page=100", paginate=True)
    reviews = github_api(f"{endpoint}/reviews?per_page=100", paginate=True)
    fields = ("id", "body", "html_url", "created_at", "updated_at", "submitted_at", "state", "commit_id")

    def compact(item: dict) -> dict:
        return {
            **{key: item[key] for key in fields if key in item},
            "author": (item.get("user") or {}).get("login"),
        }

    return {
        "repository": repo,
        "number": pr["number"],
        "head": pr["head"]["sha"],
        "base": pr["base"]["sha"],
        "base_ref": pr["base"]["ref"],
        "title": pr["title"],
        "body": pr["body"],
        "author": pr["user"]["login"],
        "comments": [compact(item) for item in comments],
        "reviews": [compact(item) for item in reviews],
        "threads": review_threads(repo, pr["number"]),
    }


def prepare(event_name: str, event: dict, repo: str, destination: Path, outputs: Path) -> None:
    """Produce a bounded snapshot and shell-safe workflow outputs, or skip the event."""
    pr = review_target(event_name, event, repo)
    if pr is None:
        return
    snapshot = discussion_snapshot(repo, pr)
    contents = json.dumps(snapshot, ensure_ascii=False)
    if len(contents.encode()) > MAX_CONTEXT_BYTES:
        raise ValueError("PR discussion exceeds 8 MiB; refusing an incomplete review context")
    destination.write_text(contents)
    # Do not embed ref names or discussion text into workflow shell expressions.
    with outputs.open("a") as stream:
        stream.write(
            f"number={pr['number']}\nhead={pr['head']['sha']}\nbase={pr['base']['sha']}\nready=true\n"
        )


if __name__ == "__main__":
    prepare(
        os.environ["GITHUB_EVENT_NAME"],
        json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()),
        os.environ["GITHUB_REPOSITORY"],
        Path(sys.argv[1]),
        Path(os.environ["GITHUB_OUTPUT"]),
    )
