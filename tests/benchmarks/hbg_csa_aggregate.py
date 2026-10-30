# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import hashlib
import json
import math
import random
import statistics as st
import subprocess
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Verify and aggregate pinned CSA version evidence (no NPU required)."
)
parser.add_argument("--evidence", type=Path, required=True)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--device", required=True, help="Physical device recorded in the evidence; analysis only")
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert args.repeats >= 2
e = args.evidence
repo = args.repo
versions = {"before": "d5ee5fd", "pre-unify": "12460f9", "current": "6850dc0"}
expected = {}
for v, rev in versions.items():
    paths = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", rev, "python/pypto"], text=True
    ).splitlines()
    expected[v] = {
        p.removeprefix("python/"): hashlib.sha256(
            subprocess.check_output(["git", "-C", str(repo), "show", f"{rev}:{p}"])
        ).hexdigest()
        for p in paths
        if p.endswith(".py")
    }
complete = []
missing = []
records = {}
pids = set()
native = None
for h in (8192, 131072):
    for q in (0, 1):
        for repeat in range(1, args.repeats + 1):
            for v, rev in versions.items():
                name = f"h{h}-q{q}-r{repeat}-{v}"
                p = e / name
                if not (p / "report.json").exists():
                    missing.append(name)
                    continue
                report = json.loads((p / "report.json").read_text())
                assert report["status"] == "MEASURED"
                m = json.loads((p / "version-manifest.json").read_text())
                assert m["revision"].startswith(rev) and m["version"] == v
                assert m["python_sources"] == expected[v]
                assert (
                    m["runtime_pin"]
                    == subprocess.check_output(
                        ["git", "-C", str(repo), "rev-parse", f"{rev}:runtime"], text=True
                    ).strip()
                )
                assert (
                    report["history"] == h
                    and report["batch"] == 4
                    and report["layer_index"] == 2
                    and report["seed"] == 1024
                )
                assert (
                    report["execution_config"]["path"] == "kernel_eager"
                    and report["execution_config"]["pto_runtime"] == "host_build_graph"
                )
                init_path = Path(m["loaded_modules"]["pypto"]["path"])
                assert init_path.as_posix().endswith("/pypto/__init__.py")
                source_root = init_path.parents[1]
                for x in m["loaded_modules"].values():
                    path = Path(x["path"])
                    if path.suffix == ".py":
                        assert path.is_relative_to(source_root), (path, source_root)
                        assert x["sha256"] == expected[v][str(path.relative_to(source_root))]
                ident = json.loads((p / "identity.json").read_text())
                assert ident["environment"]["ASCEND_RT_VISIBLE_DEVICES"] == args.device
                assert int(ident["environment"]["TASK_QUEUE_ENABLE"]) == q
                assert ident["pid"] not in pids
                pids.add(ident["pid"])
                assert ident["simpler_log_level"] == 40
                libs = {n: x for n, x in ident["modules"].items() if n != "pypto"}
                if native is None:
                    native = libs
                assert libs == native
                log = p.with_suffix(".log").read_text()
                assert "[TIMING]" not in log and "[STRACE]" not in log
                checks = json.loads((p / "pto-checks.json").read_text())["checks"]
                cpu = json.loads((p / "pto-lifecycle.json").read_text())
                assert len(checks) == 12 and len(cpu) == 12
                assert {(r["batch"], r["stage"]) for r in cpu} == {
                    (b, s) for b in range(6) for s in ("current_a", "current_b")
                }
                assert all(r["calls"] == 8 and math.isfinite(r["cpu_us"]) and r["cpu_us"] > 0 for r in cpu)
                for c in checks + [r["checks"] for r in cpu]:
                    assert all(z["status"] == "PASS" for z in c["guards"].values())
                    assert all(z["status"] == "PASS" for z in c["comparison"].values())
                    assert c["hashes"] == checks[0]["hashes"]
                rows = json.loads((p / "pto-host.json").read_text())
                assert len(rows) == 384
                assert {(r["batch"], r["stage"], r["sample"]) for r in rows} == {
                    (b, s, i) for b in range(6) for s in ("current_a", "current_b") for i in range(32)
                }
                assert all(math.isfinite(r["host_us"]) and r["host_us"] > 0 for r in rows)
                groups = defaultdict(list)
                for r in rows:
                    groups[r["batch"], r["stage"]].append(r["host_us"])
                assert set(groups) == {(b, s) for b in range(6) for s in ("current_a", "current_b")}
                assert all(len(x) == 32 for x in groups.values())
                med = {k: st.median(x) for k, x in groups.items()}
                base = [(med[b, "current_a"] + med[b, "current_b"]) / 2 for b in range(6)]
                records[h, q, repeat, v] = {
                    "median": st.median(base),
                    "aa": max(abs(med[b, "current_a"] - med[b, "current_b"]) for b in range(6)),
                    "hashes": checks[0]["hashes"],
                }
                complete.append(name)
assert not missing, f"Incomplete comparison: {missing}"
result = {}
for h in (8192, 131072):
    for q in (0, 1):
        data = {v: [records[h, q, r, v] for r in range(1, args.repeats + 1)] for v in versions}
        for r in range(1, args.repeats + 1):
            assert len({json.dumps(records[h, q, r, v]["hashes"], sort_keys=True) for v in versions}) == 1
        med = {v: st.median(x["median"] for x in xs) / 1000 for v, xs in data.items()}
        noise = {
            v: max([x["aa"] for x in xs] + [max(x["median"] for x in xs) - min(x["median"] for x in xs)])
            / 1000
            for v, xs in data.items()
        }
        comparisons = {}
        for a, b in [("before", "current"), ("pre-unify", "current")]:
            d = [(x["median"] - y["median"]) / 1000 for x, y in zip(data[a], data[b])]
            rng = random.Random(4242)
            boot = sorted(st.median(rng.choices(d, k=args.repeats)) for _ in range(50000))
            ci = [boot[1249], boot[48749]]
            threshold = max(noise[a], noise[b])
            comparisons[a + "->" + b] = {
                "paired_saved_ms": st.median(d),
                "process_deltas_ms": d,
                "ci95_ms": ci,
                "noise_ms": threshold,
                "passes": min(d) > 0 and ci[0] > threshold,
            }
        result[f"h{h}-q{q}"] = {"median_ms": med, "noise_ms": noise, "comparisons": comparisons}
out = {
    "complete_count": len(complete),
    "missing": missing,
    "scope": f"{args.repeats} complete paired repeats on physical {args.device}",
    "configs": result,
}
args.output.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
