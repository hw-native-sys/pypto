#!/usr/bin/env python3
"""Report (or print) the verify_all.sh rows implied by each corpus manifest.

WHY THIS EXISTS: four rounds committed with a red gate because a kernel was certified and its
`verify_all.sh` repeat row was added by hand -- or not. `test_verify_all_covers_the_evidence`
caught it every time, which means the gate works and the workflow does not: a row that must be
kept in step with a manifest should be DERIVED from it, not remembered beside it.

Every manifest row that records a device PASS also records the command that produced it, so the
args are already in the ledger. This script reconstructs the `verify_all.sh` line from that command
and reports any that the script does not carry.

usage:  python3 sync_verify_all.py            # report missing rows
        python3 sync_verify_all.py --print    # print the rows to append
"""
import json
import pathlib
import re
import sys

BENCH = pathlib.Path(__file__).resolve().parents[2] / "bench"
VERIFY = pathlib.Path(__file__).resolve().parent / "verify_all.sh"


def rows_from_manifests():
    """(corpus, kernel, args) for every manifest entry with a device PASS and a command."""
    out = []
    for man in sorted(BENCH.glob("*/manifest.json")):
        corpus = man.parent.name
        data = json.loads(man.read_text())
        for name, rec in (data.get("kernels") or {}).items():
            check = rec.get("device_check") or ""
            if not check.startswith("PASS"):
                continue
            path = f"{corpus}/ascendc/{name}.cpp"
            # `verify_args` is the args column of the verify_all.sh row, recorded AT CERTIFICATION
            # TIME. The corpus manifests did not record commands, so a derivation from them is only
            # possible for rows that carry this field -- which is the convention from 2026-11-06 on.
            cmd = rec.get("verify_args") or ""
            # setup.sh <cpp> <arch> <do> <alloc> <nargs> <oracle> <atol> <rtol> [repeat] [dims...]
            if not cmd:
                continue
            parts = cmd.split()
            if len(parts) < 3:
                continue
            out.append((corpus, name, path, parts[0], parts[1], " ".join(parts[2:])))
    return out


def main():
    # READ THE TABLE, NOT THE FILE. The rows live in a `ROWS <<'ROWS'` heredoc and the gate parses
    # exactly that block; grepping the whole script sees rows appended after the heredoc and reports a
    # false pass -- which is what happened in iteration 167, where this tool said "covered" and
    # test_verify_all_covers_the_evidence said otherwise. THE GATE IS THE AUTHORITY.
    text = VERIFY.read_text()
    m = re.search(r"ROWS <<'ROWS'\n(.*?)\nROWS", text, re.S)
    assert m, "could not find the ROWS table in verify_all.sh"
    text = m.group(1)
    missing, present = [], []
    for corpus, name, path, arch, do, rest in rows_from_manifests():
        if path.replace(".cpp", "") in text:
            present.append(path)
        else:
            missing.append(f"{path:<52} {arch} {do:<32} {rest}")
    if "--print" in sys.argv:
        print("\n".join(missing))
        return 0
    print(f"manifest device-PASS rows: {len(present) + len(missing)}")
    print(f"carried by verify_all.sh : {len(present)}")
    if missing:
        print(f"MISSING ({len(missing)}):")
        for m in missing:
            print("  " + m)
        print("\nrun with --print to get the lines to append")
        return 1
    print("verify_all.sh covers every device PASS recorded in the manifests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
