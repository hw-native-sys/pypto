# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

FROM node:24-bookworm-slim@sha256:0e0ff40c39bc087845bfb27465a0df4ea419520094bc35842ff83dd8cbe6f9b6

ARG CODEX_VERSION=0.156.1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends bubblewrap ca-certificates git \
    && npm install --global "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1003 codex-review \
    && useradd --uid 1002 --gid 1003 --create-home --home-dir /home/codex-review codex-review

RUN apt-get update \
    && apt-get install --yes --no-install-recommends inotify-tools \
    && rm -rf /var/lib/apt/lists/*

COPY codex-config.toml /usr/local/share/codex-review/config.toml
COPY entrypoint.sh /usr/local/bin/codex-review-entrypoint
COPY proxy-sidecar.mjs /usr/local/share/codex-review/proxy-sidecar.mjs
COPY proxy_relay.py /usr/local/share/codex-review/proxy_relay.py

RUN chmod 0555 /usr/local/bin/codex-review-entrypoint \
    /usr/local/share/codex-review/proxy-sidecar.mjs \
    && chmod 0444 /usr/local/share/codex-review/config.toml \
    /usr/local/share/codex-review/proxy_relay.py

USER 1002:1003
WORKDIR /workspace

ENTRYPOINT ["/bin/bash", "/usr/local/bin/codex-review-entrypoint"]
