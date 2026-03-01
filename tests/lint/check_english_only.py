# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Script to check that all source files and documentation are in English only.
Detects non-English text (e.g., Chinese, Japanese, Korean, etc.) in source files.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Default excluded directories (can be overridden via --exclude)
DEFAULT_EXCLUDED_PATTERNS = ["3rdparty", "reference", "docs/zh-cn", "README.zh-CN.md", "README.md"]


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


def contains_non_english(text: str) -> tuple[bool, list[tuple[int, str]]]:
    """
    Check if text contains non-English characters.

    Returns:
        Tuple of (has_non_english, list of (line_number, non-english text) tuples)
    """
    # Unicode ranges for common non-English scripts:
    # - Chinese (CJK Unified Ideographs): \u4e00-\u9fff
    # - Japanese Hiragana: \u3040-\u309f
    # - Japanese Katakana: \u30a0-\u30ff
    # - Korean Hangul: \uac00-\ud7af
    # - Cyrillic: \u0400-\u04ff
    # - Arabic: \u0600-\u06ff
    # - Hebrew: \u0590-\u05ff
    # - Thai: \u0e00-\u0e7f
    # - CJK Symbols and Punctuation: \u3000-\u303f (ideographic period, comma, brackets)
    # - Full-width Latin and Punctuation: \uff01-\uff5e (full-width comma, colon, parens)
    non_english_pattern = re.compile(
        r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af"
        r"\u0400-\u04ff\u0600-\u06ff\u0590-\u05ff\u0e00-\u0e7f"
        r"\u3000-\u303f\uff01-\uff5e]+"
    )

    violations = []
    lines = text.split("\n")

    for i, line in enumerate(lines, 1):
        matches = non_english_pattern.findall(line)
        if matches:
            # Join all non-English text found in this line
            non_english_text = ", ".join(matches)
            violations.append((i, non_english_text))

    return bool(violations), violations


def check_file_english_only(file_path: Path) -> tuple[bool, list[tuple[int, str]]]:
    """
    Check if a file contains only English text.

    Returns:
        Tuple of (is_english_only, list of (line_number, non_english_text) violations)
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return True, []  # Skip files we can't read

    has_non_english, violations = contains_non_english(content)

    return not has_non_english, violations


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check that all source files and documentation are in English only"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to git repository (default: current directory)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Additional directory patterns to exclude (can be specified multiple times)",
    )

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

    # Combine default exclusions with user-provided ones
    excluded_patterns = DEFAULT_EXCLUDED_PATTERNS.copy()
    if args.exclude:
        excluded_patterns.extend(args.exclude)

    # Filter out excluded directories
    filtered_files = []
    for f in all_files:
        relative_path = f.relative_to(root_path)
        if not any(str(relative_path).startswith(pattern) for pattern in excluded_patterns):
            filtered_files.append(f)

    # Filter to only source and documentation files
    source_extensions = {".py", ".pyi", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx", ".md"}
    files_to_check = [f for f in filtered_files if f.suffix in source_extensions]

    if not files_to_check:
        print("No source files found to check")
        return 0

    print(f"Checking {len(files_to_check)} file(s) for English-only content...")
    if excluded_patterns:
        print(f"Excluded patterns: {', '.join(excluded_patterns)}")

    failed_files = []
    passed_files = []

    for file_path in files_to_check:
        is_english_only, violations = check_file_english_only(file_path)

        if is_english_only:
            passed_files.append(file_path)
            if args.verbose:
                print(f"✓ {file_path.relative_to(root_path)}")
        else:
            failed_files.append((file_path, violations))
            # Show first 5 violations in IDE-friendly format: file:line message
            for line_num, non_english_text in violations[:5]:
                print(f"✗ {file_path.relative_to(root_path)}:{line_num} {non_english_text}")
            if len(violations) > 5:
                print(f"  ... and {len(violations) - 5} more line(s) in {file_path.relative_to(root_path)}")

    print()
    print(f"Results: {len(passed_files)} passed, {len(failed_files)} failed")

    if failed_files:
        print("\nFiles with non-English content:")
        for file_path, _ in failed_files:
            print(f"  - {file_path.relative_to(root_path)}")
        print(
            "\n⚠ Please ensure all source files and documentation are written in English "
            "to maintain consistency and accessibility for all contributors."
        )
        print("\nTip: Use --exclude to exclude specific directories (e.g., --exclude docs/zh-cn)")
        return 1

    print("\n✓ All source files and documentation are in English!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
