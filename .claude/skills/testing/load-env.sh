# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Load resource limits from the primary checkout so linked worktrees share the
# same machine profile. Unclassified machines use conservative defaults.
PYPTO_GIT_COMMON_DIR=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)
PYPTO_PRIMARY_WORKTREE=$(dirname "$PYPTO_GIT_COMMON_DIR")
PYPTO_TESTING_ENV="$PYPTO_PRIMARY_WORKTREE/.claude/skills/testing/testing.env"

if [ -f "$PYPTO_TESTING_ENV" ]; then
    source "$PYPTO_TESTING_ENV" || return $?
fi

export PYPTO_MACHINE_PROFILE="${PYPTO_MACHINE_PROFILE:-unclassified}"
export PYPTO_BUILD_JOBS="${PYPTO_BUILD_JOBS:-2}"
export PYPTO_TEST_JOBS="${PYPTO_TEST_JOBS:-2}"

case "$PYPTO_BUILD_JOBS" in
    '' | *[!0-9]*)
        echo "PYPTO_BUILD_JOBS must be a positive decimal integer, got: $PYPTO_BUILD_JOBS" >&2
        unset PYPTO_GIT_COMMON_DIR PYPTO_PRIMARY_WORKTREE PYPTO_TESTING_ENV
        return 1
        ;;
esac
if [ "$PYPTO_BUILD_JOBS" -eq 0 ]; then
    echo "PYPTO_BUILD_JOBS must be greater than zero" >&2
    unset PYPTO_GIT_COMMON_DIR PYPTO_PRIMARY_WORKTREE PYPTO_TESTING_ENV
    return 1
fi

case "$PYPTO_TEST_JOBS" in
    '' | *[!0-9]*)
        echo "PYPTO_TEST_JOBS must be a positive decimal integer, got: $PYPTO_TEST_JOBS" >&2
        unset PYPTO_GIT_COMMON_DIR PYPTO_PRIMARY_WORKTREE PYPTO_TESTING_ENV
        return 1
        ;;
esac
if [ "$PYPTO_TEST_JOBS" -eq 0 ]; then
    echo "PYPTO_TEST_JOBS must be greater than zero" >&2
    unset PYPTO_GIT_COMMON_DIR PYPTO_PRIMARY_WORKTREE PYPTO_TESTING_ENV
    return 1
fi

export CMAKE_BUILD_PARALLEL_LEVEL="$PYPTO_BUILD_JOBS"
export MAKEFLAGS="-j$PYPTO_BUILD_JOBS"
export MAX_JOBS="$PYPTO_BUILD_JOBS"
export PYTEST_XDIST_AUTO_NUM_WORKERS="$PYPTO_TEST_JOBS"

unset PYPTO_GIT_COMMON_DIR PYPTO_PRIMARY_WORKTREE PYPTO_TESTING_ENV
