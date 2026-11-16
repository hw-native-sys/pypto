#!/usr/bin/env python3
"""Check that what the READMEs promise exists in the code.

Documentation drifts silently: a flag is renamed, an env var is dropped, and the
README keeps promising it. This checks three things and exits non-zero on a
mismatch, so a PR cannot merge a claim the code does not back:

  1. every `ds4-rs-ascend <subcommand>` in the docs is a real subcommand;
  2. every `--flag` attributed to ds4-rs-ascend exists in its CLI parser;
  3. every DS4_*/PTOAS_* environment variable named in the docs is read by code.

It does NOT check prose, numbers, or whether a demo recording is current --
those need a human or a recorder.
"""
import re, subprocess, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "rust" / "README.md"
DS4 = ROOT / "rust" / "ds4-rs-ascend"
CLI = DS4 / "crates" / "controller" / "src" / "cli.rs"

def read(p):
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

def main():
    docs = [README, DS4 / "README.md"] + sorted((ROOT / "rust" / "crates").glob("*/README.md"))
    text = "\n".join(read(p) for p in docs)
    problems = []

    cli = read(CLI)
    for sub in sorted(set(re.findall(r"ds4-rs-ascend\s+([a-z]+)", text))):
        if sub in {"demo"}:
            continue
        if f'"{sub}"' not in cli:
            problems.append(f"documented subcommand `ds4-rs-ascend {sub}` is not in cli.rs")

    # Only flags written on a line that names ds4-rs-ascend belong to it; the
    # other four utilities have their own parsers and their flags are not here.
    # A line that also invokes cargo (e.g. `cd ds4-rs-ascend && cargo build
    # --release`) carries cargo's flags, not the tool's.
    ds4_lines = "\n".join(
        l for l in text.splitlines()
        if "ds4-rs-ascend" in l and "cargo " not in l
    )
    for flag in sorted(set(re.findall(r"(--[a-z][a-z-]+)", ds4_lines))):
        if flag in {"--version", "--help"}:
            continue
        if flag not in cli:
            problems.append(f"documented ds4-rs-ascend flag `{flag}` is not in cli.rs")

    code = "\n".join(read(p) for p in (ROOT / "rust").rglob("*.rs"))
    for env in sorted(set(re.findall(r"\b(DS4_[A-Z_]+|PTOAS_[A-Z_]+)\b", text))):
        if env not in code:
            problems.append(f"documented env var `{env}` is not read anywhere in ds4-rs-ascend")

    if problems:
        print("documentation and code disagree:")
        for p in problems:
            print("  -", p)
        return 1
    print("docs/code consistency: ok")
    return 0

if __name__ == "__main__":
    sys.exit(main())
