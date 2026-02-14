# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Script to check copyright headers in source files.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Define the expected headers
PY_HEADER = """
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""".strip()

C_HEADER = """
/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
""".strip()

# File extension to header mapping
FILE_TYPE_HEADERS = {
    # Python files
    ".py": PY_HEADER,
    ".pyi": PY_HEADER,
    # C/C++ files
    ".c": C_HEADER,
    ".cpp": C_HEADER,
    ".cc": C_HEADER,
    ".cxx": C_HEADER,
    ".h": C_HEADER,
    ".hpp": C_HEADER,
    ".hxx": C_HEADER,
    # CMake files
    ".cmake": PY_HEADER,
    # Shell scripts
    ".sh": PY_HEADER,
    # Configuration files
    ".toml": PY_HEADER,
}

# Files matched by name (not extension)
FILE_NAME_HEADERS = {
    "CMakeLists.txt": PY_HEADER,
    ".clang-format": PY_HEADER,
    ".clang-tidy": PY_HEADER,
}


def get_git_tracked_files(root_dir: Path) -> list[Path]:
    """Get list of files tracked by git."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        files = []
        for line in result.stdout.strip().split("\n"):
            if line:
                file_path = root_dir / line
                if file_path.is_file():
                    files.append(file_path)
        return files
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to get git tracked files: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: git command not found", file=sys.stderr)
        sys.exit(1)


def check_file_header(file_path: Path) -> tuple[bool, str]:
    """
    Check if a file has the correct header.

    Returns:
        Tuple of (has_correct_header, error_message)
    """
    ext = file_path.suffix
    name = file_path.name

    # Check by filename first, then by extension
    if name in FILE_NAME_HEADERS:
        expected_header = FILE_NAME_HEADERS[name]
    elif ext in FILE_TYPE_HEADERS:
        expected_header = FILE_TYPE_HEADERS[ext]
    else:
        return True, ""  # Skip files we don't know how to check

    expected_lines = expected_header.strip().split("\n")

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError) as e:
        return False, f"Error reading file: {e}"

    if not content:
        return False, "File is empty"

    actual_lines = content.split("\n")

    # Check if file has enough lines
    if len(actual_lines) < len(expected_lines):
        return (
            False,
            f"File has {len(actual_lines)} lines, but header requires {len(expected_lines)} lines",
        )

    # Check each line of the header
    for i, expected_line in enumerate(expected_lines):
        actual_line = actual_lines[i]
        if actual_line != expected_line:
            return (
                False,
                f"Line {i + 1} doesn't match:\n  Expected: {expected_line}\n  Got:      {actual_line}",
            )

    return True, ""


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check copyright headers in git-tracked source files")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to git repository (default: current directory)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    root_path = Path(args.path).resolve()

    if not root_path.exists():
        print(f"Error: Path '{root_path}' does not exist", file=sys.stderr)
        return 1

    if not (root_path / ".git").exists():
        print(f"Error: '{root_path}' is not a git repository", file=sys.stderr)
        return 1

    # Get all git-tracked files
    all_files = get_git_tracked_files(root_path)

    # Filter out excluded directories
    excluded_patterns = ["reference"]
    filtered_files = []
    for f in all_files:
        relative_path = f.relative_to(root_path)
        if not any(str(relative_path).startswith(pattern) for pattern in excluded_patterns):
            filtered_files.append(f)

    # Filter to only files we know how to check
    files_to_check = [
        f for f in filtered_files if f.suffix in FILE_TYPE_HEADERS or f.name in FILE_NAME_HEADERS
    ]

    if not files_to_check:
        print("No source files found to check")
        return 0

    print(f"Checking {len(files_to_check)} file(s)...")

    failed_files = []
    passed_files = []

    for file_path in files_to_check:
        has_header, error_msg = check_file_header(file_path)

        if has_header:
            passed_files.append(file_path)
            if args.verbose:
                print(f"✓ {file_path.relative_to(root_path)}")
        else:
            failed_files.append((file_path, error_msg))
            print(f"✗ {file_path.relative_to(root_path)}")
            if error_msg:
                print(f"  {error_msg}")

    print()
    print(f"Results: {len(passed_files)} passed, {len(failed_files)} failed")

    if failed_files:
        print("\nFiles with incorrect or missing headers:")
        for file_path, _ in failed_files:
            print(f"  - {file_path.relative_to(root_path)}")
        return 1

    print("\n✓ All files have correct headers!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
