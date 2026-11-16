#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check tile_std against the nightlies its build.rs date gates promise to support.

tile_std is a #![no_core] crate wired to compiler internals, and build.rs keeps
it building across nightlies by reading rustc's commit date and setting one cfg
per compiler change, from its `GATES` table of (cfg name, first date). A gate
with the wrong date breaks every nightly between the right date and the wrong
one, and checking only the pinned and the latest nightly never sees that.

Default run, one toolchain at a time:
  * the pinned nightly (rust-toolchain.toml): `cargo build -p tile_std`
  * for every gate, the nightly just before and just after its date:
    `cargo check -p tile_std`, confirming the gate is off, then on
  * the latest nightly: `cargo check -p tile_std`

Bisect a breakage to the nightly where it starts, optionally tracking one
error message so that unrelated breakages in the same range do not interfere:
  nightly_drift_check.py --bisect 2026-07-14 2026-09-16
  nightly_drift_check.py --bisect 2025-08-04 2025-09-01 --signature rustc_coherence_is_core

Toolchains this script installs are uninstalled after their check (disk on
shared machines is tight); toolchains already present are left alone.

Usage:
  python3 scripts/nightly_drift_check.py [--tree DIR] [--offline]
                                         [--only pinned|gates|latest]
                                         [--bisect GOOD BAD] [--keep-toolchains]
"""

import argparse
import datetime as dt
import os
import re
import subprocess
import sys

import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_RE = re.compile(r'\("(\w+)",\s*\((\d{4}),\s*(\d+),\s*(\d+)\)\)')
ERROR_RE = re.compile(r"^(error(?:\[E\d+\])?: .*)$", re.M)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def sh(cmd, **kw):
    # `check=False` explicitly: every caller reads the returncode itself,
    # because a nonzero rustc is the SIGNAL here and not an error.
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)


def gates(tree):
    """Read the (cfg name, date) table `GATES` from tile_std's build.rs."""
    src = open(os.path.join(tree, "crates/tile_std/build.rs")).read()
    table = src[src.index("const GATES") :]
    table = table[: table.index("];")]
    found = [
        (m.group(1), dt.date(int(m.group(2)), int(m.group(3)), int(m.group(4))))
        for m in GATE_RE.finditer(table)
    ]
    if not found:
        sys.exit("found no gates in crates/tile_std/build.rs (expected a `const GATES` table)")
    return sorted(found, key=lambda g: g[1])


def installed():
    out = sh(["rustup", "toolchain", "list"]).stdout
    return {line.split()[0] for line in out.splitlines() if line.strip()}


def ensure(tc, keep, have):
    """Install `tc` if needed. Returns (ok, installed_by_us)."""
    host = sh(["rustc", "-vV"]).stdout
    triple = re.search(r"host: (\S+)", host).group(1)
    if f"{tc}-{triple}" in have or tc in have:
        return True, False
    r = sh(["rustup", "toolchain", "install", tc, "--profile", "minimal", "--no-self-update"])
    return r.returncode == 0, r.returncode == 0 and not keep


def commit_date(tc):
    out = sh(["rustc", f"+{tc}", "-vV"]).stdout
    m = re.search(r"commit-date: (\d{4})-(\d{2})-(\d{2})", out)
    return dt.date(*map(int, m.groups())) if m else None


def run_cargo(tc, tree, build, offline):
    target = os.path.join(tree, "target", "nightly-drift", tc)
    cmd = ["cargo", f"+{tc}", "build" if build else "check", "-p", "tile_std"]
    if offline:
        cmd.append("--offline")
    env = dict(os.environ, CARGO_TARGET_DIR=target, RUSTFLAGS="")
    r = sh(cmd, cwd=tree, env=env)
    text = ANSI.sub("", r.stderr)
    errors = sorted(set(e for e in ERROR_RE.findall(text) if not e.startswith("error: could not compile")))
    return r.returncode == 0, errors


def check_one(tc, tree, build, offline, keep, have, expect=None):
    ok_install, ours = ensure(tc, keep, have)
    if not ok_install:
        return {"tc": tc, "status": "NO TOOLCHAIN", "errors": [], "date": None}
    try:
        date = commit_date(tc)
        ok, errors = run_cargo(tc, tree, build, offline)
        status = "ok" if ok else "FAIL"
        if expect is not None and date is not None:
            name, gate_date, want_on = expect
            if (date >= gate_date) != want_on:
                status = "WRONG SIDE"
                errors = [
                    f"commit-date {date} is on the wrong side of {name} ({gate_date}); pick another nightly"
                ] + errors
        return {"tc": tc, "status": status, "errors": errors, "date": date}
    finally:
        if ours:
            sh(["rustup", "toolchain", "uninstall", tc])


def active(date, all_gates):
    return [name for name, d in all_gates if date is not None and date >= d]


def nightly(day):
    return f"nightly-{day.isoformat()}"


def report(rows, all_gates):
    width = max(len(r["tc"]) for r in rows)
    for r in rows:
        label = r.get("label", "")
        cfgs = ",".join(active(r["date"], all_gates)) or "-"
        print(f"{r['tc']:<{width}}  {str(r['date']):<10}  {r['status']:<11} {label:<42} cfgs: {cfgs}")
        for e in r["errors"][:8]:
            print(f"      {e}")
        if len(r["errors"]) > 8:
            print(f"      ... {len(r['errors']) - 8} more distinct errors")


def bisect(good, bad, tree, offline, keep, signature=None):
    """Find where a predicate flips between two nightly dates.

    Without `signature` the predicate is "the check fails". With it, the
    predicate is "some error contains `signature`", which isolates one compiler
    change when several break the same range. The endpoints are measured first
    and must differ.
    """
    have = installed()

    def measure(day):
        # Step forward over days with no published nightly.
        for _ in range(4):
            r = check_one(nightly(day), tree, False, offline, keep, have)
            if r["status"] != "NO TOOLCHAIN":
                hit = any(signature in e for e in r["errors"]) if signature else r["status"] != "ok"
                print(
                    f"  {nightly(day)} (commit {r['date']}): {'HIT' if hit else 'clear'} [{r['status']}]",
                    flush=True,
                )
                return day, r, hit
            day += dt.timedelta(days=1)
        sys.exit(f"no nightly published near {day}")

    lo, rlo, hit_lo = measure(good)
    hi, rhi, hit_hi = measure(bad)
    if hit_lo == hit_hi:
        sys.exit(f"both endpoints are {'HIT' if hit_lo else 'clear'}; nothing to bisect")
    while (hi - lo).days > 1:
        mid = lo + dt.timedelta(days=(hi - lo).days // 2)
        day, r, hit = measure(mid)
        if day >= hi:
            break
        if hit == hit_lo:
            lo, rlo = day, r
        else:
            hi, rhi = day, r
    what = f"errors containing {signature!r}" if signature else "failures"
    change = "appear" if hit_hi else "disappear"
    print(
        f"\n{what} {change} between {nightly(lo)} (commit {rlo['date']}) "
        f"and {nightly(hi)} (commit {rhi['date']})"
    )
    print(
        f"a build.rs gate for this change should read: "
        f"d >= ({rhi['date'].year}, {rhi['date'].month}, {rhi['date'].day})"
    )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=ROOT, help="tree whose root workspace contains tile_std")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--only", choices=["pinned", "gates", "latest"])
    ap.add_argument("--bisect", nargs=2, metavar=("FROM", "TO"))
    ap.add_argument("--signature", help="with --bisect: track one error message instead of pass/fail")
    ap.add_argument("--keep-toolchains", action="store_true")
    a = ap.parse_args()
    tree = os.path.abspath(a.tree)

    if a.bisect:
        good, bad = (dt.date.fromisoformat(x) for x in a.bisect)
        return bisect(good, bad, tree, a.offline, a.keep_toolchains, a.signature)

    all_gates = gates(tree)
    pinned = tomllib.load(open(os.path.join(tree, "rust-toolchain.toml"), "rb"))["toolchain"]["channel"]
    have = installed()
    rows = []

    if a.only in (None, "pinned"):
        r = check_one(pinned, tree, True, a.offline, a.keep_toolchains, have)
        r["label"] = "pinned (build)"
        rows.append(r)
        print(f"{pinned}: {r['status']}", flush=True)

    if a.only in (None, "gates"):
        # A nightly named for day D carries the previous day's commit date.
        for name, gate_date in all_gates:
            # nightly-(gate) usually carries commit date gate-1 (off); nightly-(gate+1)
            # carries the gate date itself (on). WRONG SIDE flags a nightly whose
            # commit date does not fall where expected.
            for side, day, want_on in (
                ("before", gate_date, False),
                ("after", gate_date + dt.timedelta(days=1), True),
            ):
                tc = nightly(day)
                r = check_one(tc, tree, False, a.offline, a.keep_toolchains, have, (name, gate_date, want_on))
                r["label"] = f"{name} {side} {gate_date}"
                rows.append(r)
                print(f"{tc}: {r['status']} ({r['label']})", flush=True)

    if a.only in (None, "latest"):
        # The newest published nightly, by date. The shared `nightly` channel is
        # never updated: other work on the machine may depend on it.
        day = dt.date.today() + dt.timedelta(days=1)
        for _ in range(7):
            r = check_one(nightly(day), tree, False, a.offline, a.keep_toolchains, have)
            if r["status"] != "NO TOOLCHAIN":
                break
            day -= dt.timedelta(days=1)
        r["label"] = "latest"
        rows.append(r)
        print(f"{r['tc']}: {r['status']} (latest)", flush=True)

    print()
    print("gates read from build.rs: " + ", ".join(f"{n} >= {d}" for n, d in all_gates))
    report(rows, all_gates)
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"\n{len(rows) - len(bad)} of {len(rows)} toolchains pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
