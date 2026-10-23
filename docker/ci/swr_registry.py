# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Promote SWR manifests without downloading layers or deleting shared digests."""

import base64
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
OCI_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"


@dataclass(frozen=True)
class Manifest:
    content: bytes
    media_type: str
    digest: str
    config_digest: str


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        # Neither manifests nor auth need redirects. Never forward credentials.
        return None


class Registry:
    """Access a single SWR repository directly, ignoring host proxy variables."""

    def __init__(self, registry: str, repository: str, username: str, password: str) -> None:
        self.base = f"https://{registry}"
        self.repository = repository
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        query = urllib.parse.urlencode(
            {"service": "dockyard", "scope": f"repository:{repository}:pull,push,delete"}
        )
        basic = base64.b64encode(f"{username}:{password}".encode()).decode()
        request = urllib.request.Request(
            f"{self.base}/swr/auth/v2/registry/auth/?{query}",
            headers={"Authorization": f"Basic {basic}"},
        )
        with self.opener.open(request, timeout=60) as response:
            auth = json.load(response)
        self.token: str = auth.get("token") or auth["access_token"]

    def request(
        self, method: str, tag: str, content: bytes | None = None, media_type: str = MEDIA_TYPE
    ) -> tuple[int, bytes, str]:
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
            raise ValueError("Registry operations require a tag name, never a digest")
        request = urllib.request.Request(
            f"{self.base}/v2/{self.repository}/manifests/{tag}",
            data=content,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": f"{MEDIA_TYPE}, {OCI_MEDIA_TYPE}",
                "Content-Type": media_type,
            },
        )
        try:
            with self.opener.open(request, timeout=60) as response:
                return response.status, response.read(), response.headers.get("Docker-Content-Digest", "")
        except urllib.error.HTTPError as error:
            if method == "GET" and error.code == 404:
                details = json.loads(error.read())
                codes = {item["code"] for item in details.get("errors", [])}
                if codes and codes <= {"MANIFEST_UNKNOWN", "NAME_UNKNOWN"}:
                    return 404, b"", ""
            raise RuntimeError(f"SWR {method} {tag} failed: HTTP {error.code}") from error

    def get(self, tag: str) -> Manifest | None:
        status, content, advertised = self.request("GET", tag)
        if status == 404:
            return None
        digest = f"sha256:{hashlib.sha256(content).hexdigest()}"
        if status != 200 or advertised != digest:
            raise ValueError(f"SWR manifest digest mismatch for {tag}")
        data = json.loads(content)
        if data.get("schemaVersion") != 2 or data.get("mediaType") not in (MEDIA_TYPE, OCI_MEDIA_TYPE):
            raise ValueError(f"Expected a single-platform image manifest for {tag}")
        return Manifest(content, data["mediaType"], digest, data["config"]["digest"])

    def put(self, tag: str, manifest: Manifest) -> None:
        status, _, _ = self.request("PUT", tag, manifest.content, manifest.media_type)
        if status != 201:
            raise RuntimeError(f"SWR did not accept manifest for {tag}: HTTP {status}")
        actual = self.get(tag)
        if actual is None or actual.digest != manifest.digest:
            raise ValueError(f"SWR promoted digest mismatch for {tag}")

    def delete_tag(self, tag: str) -> None:
        if self.get(tag) is None:
            return
        # OCI tag deletion (also verified on this SWR endpoint) returns 202:
        # https://github.com/opencontainers/distribution-spec/blob/main/spec.md#deleting-tags
        # Do not fall back to digest deletion: retained tags share the manifest.
        status, _, _ = self.request("DELETE", tag)
        if status != 202 or self.get(tag) is not None:
            raise RuntimeError(f"SWR did not remove temporary tag {tag}")
