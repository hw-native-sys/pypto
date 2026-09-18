#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


# Download a release asset without publishing an incomplete file.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 URL DESTINATION" >&2
  exit 2
fi
url=$1
destination=$2
download_dir=$(mktemp -d "$(dirname "$destination")/.download.XXXXXX")
trap 'rm -rf "$download_dir"' EXIT
partial="$download_dir/payload"

for attempt in 1 2 3 4; do
  status=0
  # A fresh curl process recalculates the resume offset after each interruption.
  http_status=$(curl --fail --location --continue-at - \
    --connect-timeout 15 --max-time 600 --speed-limit 1024 --speed-time 60 \
    --output "$partial" --write-out '%{http_code}' "$url") || status=$?
  # curl can treat HTTP 416 as successful when resuming. Never publish that
  # response: restart without a Range header to verify the complete payload.
  if [[ "$http_status" == 416 ]]; then
    status=22
  fi
  if [[ "$status" == 0 ]]; then
    mv -- "$partial" "$destination"
    exit 0
  fi
  if [[ "$status" == 33 || "$status" == 36 || "$http_status" == 416 ]]; then
    rm -f -- "$partial"
  fi
  echo "Release download attempt $attempt/4 failed (curl $status, HTTP $http_status)." >&2
  if [[ "$attempt" == 4 ]]; then
    exit "$status"
  fi
  sleep "$attempt"
done
