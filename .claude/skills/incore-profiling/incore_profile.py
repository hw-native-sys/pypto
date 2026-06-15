# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Export msprof op-simulator Insight traces for all PTOAS funcs in a case build.

Typical usage:

  python .claude/skills/incore-profiling/incore_profile.py \
    --build-dir build_output/Qwen3Decode_20260514_195003 --target a2a3

  python .claude/skills/incore-profiling/incore_profile.py \
    --case models/qwen3/14b/qwen3_14b_decode.py --target a2a3 \
    --task-submit --task-device auto \
    --run-env PTO2_RING_TASK_WINDOW=131072 \
    --run-env PTO2_RING_DEP_POOL=131072 \
    --run-env PTO2_RING_HEAP=536870912 \
    -- --runtime-profiling

CANN, the camodel SoC, and the compile arch are auto-resolved from --target;
override with --cann-set-env / --soc-version / --aicore-arch when needed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Locate the repo root regardless of where this script is placed.

    Walks up from `start` looking for a `.git` entry (a directory for a normal
    clone, a file for a worktree). Falls back to the script's parent directory
    when no `.git` is found. This keeps path resolution correct regardless of
    where the script is placed within the repo.
    """
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return start.parent


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _find_repo_root(Path(__file__).resolve())
SUCCESS_TEXT = "Profiling running finished. All task success."

# Maps a user-facing NPU target to its compile arch and the SoC-name keywords
# used to pick a camodel simulator directory.
TARGET_PROFILES: dict[str, dict[str, object]] = {
    "a2a3": {"aicore_arch": "dav-c220", "soc_keywords": ("910b",)},
    "a5": {"aicore_arch": "dav-c310", "soc_keywords": ("950", "a5")},
}

# The .pto-driven profiling-case generator shipped next to this script — emits a
# minimal, buildable camodel testcase (sizes from the kernel's sibling .pto). It
# replaces the heavyweight PTOAS validation-harness generator for profiling.
GEN_PROFILING_CASE = SCRIPT_DIR / "gen_profiling_case.py"


def resolve_generate_testcase(ptoas_root: Path | None) -> Path:
    """Locate the testcase generator.

    Defaults to the bundled `gen_profiling_case.py` (profiling-focused). When a
    full PTOAS source checkout is supplied via `--ptoas-root`, use its richer
    validation-harness generator instead (escape hatch for correctness golden).
    """
    if ptoas_root is not None:
        return ptoas_root / "test/npu_validation/scripts/generate_testcase.py"
    return GEN_PROFILING_CASE


def default_pto_isa_root() -> Path:
    return repo_path(os.environ.get("PTO_ISA_ROOT", str(Path.home() / "pto-isa")))


def discover_cann_set_env() -> Path | None:
    """Locate a CANN set_env.sh from the environment or standard install roots."""
    env_val = os.environ.get("CANN_SET_ENV")
    if env_val and Path(env_val).expanduser().is_file():
        return Path(env_val).expanduser()
    candidates: list[Path] = []
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        candidates += [Path(ascend_home) / "set_env.sh", Path(ascend_home).parent / "set_env.sh"]
    for root in ("/usr/local/Ascend", str(Path.home() / "Ascend"), "/opt/Ascend"):
        candidates += [
            Path(root) / "ascend-toolkit/set_env.sh",
            Path(root) / "ascend-toolkit/latest/set_env.sh",
            Path(root) / "set_env.sh",
        ]
    return next((c for c in candidates if c.is_file()), None)


def discover_camodel_socs(env: dict[str, str]) -> list[str]:
    """List simulator SoC names that ship libruntime_camodel.so in the CANN install."""
    ascend_home = Path(env.get("ASCEND_HOME_PATH", ""))
    if not ascend_home.is_dir():
        return []
    socs: set[str] = set()
    for pattern in (
        "*/simulator/*/lib/libruntime_camodel.so",
        "simulator/*/lib/libruntime_camodel.so",
        "tools/simulator/*/lib/libruntime_camodel.so",
    ):
        for lib in ascend_home.glob(pattern):
            socs.add(lib.parent.parent.name)
    return sorted(socs)


def select_soc_version(target: str, available: list[str], explicit: str | None) -> str:
    """Pick a camodel SoC for the target family, honoring an explicit override."""
    if explicit:
        return explicit
    keywords = TARGET_PROFILES[target]["soc_keywords"]
    matches = [s for s in available if any(k in s.lower() for k in keywords)]
    if not matches:
        raise StepError(
            f"no camodel SoC found for --target {target}; available: {available}. "
            f"Pass --soc-version explicitly."
        )
    if len(matches) > 1:
        log(
            f"[warn] multiple camodel SoCs match --target {target}: {matches}; "
            f"auto-selecting {matches[0]}. If the build/run fails with a SoC mismatch "
            f"or your device is a different variant (check `npu-smi info`), override "
            f"with --soc-version (e.g. --soc-version Ascend910B1)."
        )
    return matches[0]


class StepError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def source_env(set_env: str | Path | None, base_env: dict[str, str]) -> dict[str, str]:
    if not set_env:
        return base_env.copy()
    set_env_path = repo_path(set_env)
    if not set_env_path.is_file():
        raise StepError(f"CANN set_env.sh not found: {set_env_path}")
    cmd = f"source {sh_quote(str(set_env_path))} >/dev/null && env -0"
    cp = subprocess.run(
        ["bash", "-lc", cmd],
        capture_output=True,
        check=False,
    )
    if cp.returncode != 0:
        raise StepError(cp.stderr.decode("utf-8", errors="replace"))
    env = base_env.copy()
    for item in cp.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, val = item.split(b"=", 1)
        env[key.decode()] = val.decode("utf-8", errors="replace")
    return env


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    text = "$ " + " ".join(sh_quote(c) if " " in c else c for c in cmd) + "\n"
    cp = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    text += cp.stdout
    text += f"\n[exit {cp.returncode}]\n"
    if log_path:
        private_dir(log_path.parent)
        log_path.write_text(text, encoding="utf-8", errors="replace")
    if check and cp.returncode != 0:
        raise StepError(f"command failed rc={cp.returncode}: {' '.join(cmd)}")
    return cp


def build_run_command(args: argparse.Namespace, case_args: list[str]) -> list[str] | None:
    if args.run_cmd:
        if args.case:
            raise StepError("use either --run-cmd or --case, not both")
        base_cmd = args.run_cmd
    elif args.case:
        case = repo_path(args.case)
        if not case.is_file():
            raise StepError(f"case script not found: {case}")
        prefix = []
        for item in args.run_env or []:
            if "=" not in item:
                raise StepError(f"--run-env expects KEY=VALUE, got: {item}")
            prefix.append(item)
        base_cmd = " ".join(prefix + ["python", sh_quote(str(case))] + [sh_quote(x) for x in case_args])
    else:
        return None

    if args.task_submit:
        return ["task-submit", "--run", "--device", args.task_device, base_cmd]
    return ["bash", "-lc", base_cmd]


def build_output_dirs(build_output_root: Path) -> set[Path]:
    if not build_output_root.exists():
        return set()
    return {p.resolve() for p in build_output_root.iterdir() if p.is_dir()}


def looks_like_case_build(path: Path) -> bool:
    return (path / "ptoas").is_dir() or any(path.glob("next_levels/**/ptoas/*.cpp"))


def select_latest_build(build_output_root: Path, before: set[Path], start_time: float) -> Path:
    after = {p for p in build_output_dirs(build_output_root) if looks_like_case_build(p)}
    before = {p for p in before if looks_like_case_build(p)}
    new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if new_dirs:
        return new_dirs[0]
    recent = [p for p in after if p.stat().st_mtime >= start_time - 1]
    recent.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if recent:
        return recent[0]
    all_dirs = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)
    if all_dirs:
        return all_dirs[0]
    raise StepError(f"no case build output directory found under {build_output_root}")


def default_ptoas_sources(build_dir: Path, ptoas_dir: Path | None, source_glob: str | None) -> list[Path]:
    if source_glob:
        sources = sorted(repo_path(p) for p in glob_paths(source_glob))
    elif ptoas_dir:
        sources = sorted(repo_path(ptoas_dir).glob("*.cpp"))
    else:
        direct = sorted((build_dir / "ptoas").glob("*.cpp"))
        nested = sorted((build_dir / "next_levels").glob("**/ptoas/*.cpp"))
        sources = direct + [p for p in nested if p not in direct]
        if not sources:
            blocked = {"cases", "builds", "collect", "export"}
            sources = []
            for p in build_dir.glob("**/ptoas/*.cpp"):
                parts = set(p.relative_to(build_dir).parts)
                if parts & blocked:
                    continue
                if any(part.startswith("insight") for part in parts):
                    continue
                sources.append(p)
            sources.sort()
    return [p.resolve() for p in sources if p.is_file()]


def glob_paths(pattern: str) -> Iterable[Path]:
    p = Path(pattern).expanduser()
    if p.is_absolute():
        parent = Path(p.anchor)
        rel = str(p.relative_to(parent))
        yield from parent.glob(rel)
    else:
        yield from REPO_ROOT.glob(pattern)


def read_first_kernel_names(source: Path) -> list[str]:
    text = source.read_text(encoding="utf-8", errors="replace")
    names: list[str] = []
    for regex in (
        r"__global__\s+AICORE\s+void\s+([A-Za-z_]\w*)\s*\(",
        r"__global__\s+void\s+([A-Za-z_]\w*)\s*\(",
        r"\bAICORE\s+void\s+([A-Za-z_]\w*)\s*\(",
    ):
        for m in re.finditer(regex, text):
            name = m.group(1)
            if name not in names:
                names.append(name)
        if names:
            break
    return names


def detect_host_triplets(ascend_home: Path) -> list[str]:
    triplets = []
    for name in ("aarch64-linux", "x86_64-linux"):
        if (ascend_home / name).exists():
            triplets.append(name)
    return triplets


def make_ld_library_path(build_dir: Path, env: dict[str, str], soc_version: str) -> str:
    ascend_home = Path(env.get("ASCEND_HOME_PATH", ""))
    parts = [str(build_dir)]
    if ascend_home:
        for p in [ascend_home / "lib64", ascend_home / "devlib"]:
            if p.exists():
                parts.append(str(p))
        for triplet in detect_host_triplets(ascend_home):
            for p in [
                ascend_home / triplet / "devlib",
                ascend_home / triplet / "simulator" / soc_version / "lib",
            ]:
                if p.exists():
                    parts.append(str(p))
    old = env.get("LD_LIBRARY_PATH")
    if old:
        parts.append(old)
    return ":".join(parts)


def resolve_symbol(kernel_lib: Path, preferred_names: list[str]) -> tuple[str, str]:
    nm = run_cmd(["nm", "-D", str(kernel_lib)], check=True)
    symbols = []
    for line in nm.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[-2] in {"T", "W"}:
            symbols.append(fields[-1])
    candidates: list[tuple[str, str]] = []
    for sym in symbols:
        cf = run_cmd(["c++filt", sym], check=False)
        demangled = cf.stdout.strip() or sym
        candidates.append((sym, demangled))
    for name in preferred_names:
        for sym, demangled in candidates:
            # Mangled C++ kernels demangle to "name(...)"; extern "C" kernels
            # (the ptoas pure-kernel convention, and the synthesized mixed
            # dispatcher) keep the bare unmangled symbol "name".
            if demangled.startswith(name + "(") or demangled == name:
                return sym, demangled
    if len(candidates) == 1:
        return candidates[0]
    preview = "\n".join(f"  {sym} # {dem}" for sym, dem in candidates[:20])
    raise StepError(f"failed to resolve kernel symbol for {preferred_names}; candidates:\n{preview}")


def run_golden(case_dir: Path, env: dict[str, str], log_path: Path, timeout: int) -> None:
    golden = case_dir / "golden.py"
    if not golden.is_file():
        raise StepError(f"missing golden.py in {case_dir}")
    run_cmd(
        [sys.executable, str(golden)], cwd=case_dir, env=env, log_path=log_path, timeout=timeout, check=True
    )


def find_export_src(collect_out: Path) -> Path | None:
    opp_dirs = sorted(collect_out.glob("OPPROF_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for opp in opp_dirs:
        for candidate in (opp / "device0/tmp_dump", opp / "device0/dump", opp / "tmp_dump", opp / "dump"):
            if candidate.is_dir():
                maybe_copy_pc_start(candidate)
                return candidate
    for opp in opp_dirs:
        for candidate in opp.rglob("*"):
            if candidate.is_dir() and candidate.name in {"tmp_dump", "dump"}:
                maybe_copy_pc_start(candidate)
                return candidate
    return None


def maybe_copy_pc_start(export_src: Path) -> None:
    if (export_src / "pc_start_addr.txt").exists():
        return
    parent = export_src.parent
    for candidate in parent.glob("**/pc_start_addr.txt"):
        if candidate.is_file():
            shutil.copy2(candidate, export_src / "pc_start_addr.txt")
            return


def collect_artifacts(export_dir: Path) -> dict[str, object]:
    trace_jsons = sorted(export_dir.glob("**/simulator/trace.json"))
    visualize_bins = sorted(export_dir.glob("**/simulator/visualize_data.bin"))
    core_traces = sorted(export_dir.glob("**/simulator/core*/trace.json"))
    instr_csvs = sorted(export_dir.glob("**/*instr_exe*.csv"))
    return {
        "artifact_count": len(trace_jsons) + len(visualize_bins) + len(core_traces) + len(instr_csvs),
        "trace_json": str(trace_jsons[0]) if trace_jsons else "",
        "visualize_data_bin": str(visualize_bins[0]) if visualize_bins else "",
        "core_trace_count": len(core_traces),
        "instr_csv_count": len(instr_csvs),
    }


def detect_degenerate_trace(export_dir: Path) -> str | None:
    """Best-effort check: flag a near-empty trace (no compute-pipe cycles).

    A data-dependent kernel whose loop bound / work-table size is READ FROM AN
    INPUT TENSOR runs ~0 iterations because the auto-generated golden zero-fills
    integer inputs. The kernel then executes only its scalar prologue + sync
    handshakes — the CUBE pipe shows only SET_FLAG/WAIT_FLAG, never MMAD — and
    looks misleadingly free. Scans the per-core ``*_instr_exe.csv`` and warns when
    NO matmul (MMAD) ran AND the VECTOR pipe booked negligible cycles, i.e.
    neither engine did real work. (A handful of MTE / VECTOR-setup cycles always
    appear from the prologue, so this keys on MMAD presence + a small VECTOR floor
    rather than a bare zero test.) Returns ``None`` when the trace looks real or no
    CSV is present.
    """
    vec_floor = 1000  # prologue VECTOR-setup noise is ~tens of cycles, far below real work
    mmad = 0
    vec_cycles = 0
    seen_csv = False
    for csv_path in export_dir.glob("**/*instr_exe*.csv"):
        seen_csv = True
        try:
            with csv_path.open(encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    if (row.get("instr") or "").upper().startswith("MMAD"):
                        mmad += 1
                    if (row.get("pipe") or "").upper() == "VECTOR":
                        try:
                            vec_cycles += int(float(row.get("cycles") or 0))
                        except ValueError:
                            pass
        except OSError:
            continue
    if seen_csv and mmad == 0 and vec_cycles < vec_floor:
        return (
            "degenerate trace: no MMAD ran and VECTOR booked ~0 cycles, so neither "
            "engine did real compute. Either a data-dependent kernel that ran 0 "
            "iterations (auto golden zeroed its control/loop-bound input — wire real "
            "intermediates, see SKILL.md 'Caveats') or a pure data-movement kernel "
            "(e.g. a seed/memset)."
        )
    return None


def write_outputs(run_root: Path, results: list[dict[str, object]], fieldnames: list[str]) -> None:
    with (run_root / "manifest_export.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    ok = [r for r in results if r.get("status") == "exported"]
    failed = [r for r in results if r.get("status") != "exported"]
    lines = [
        f"run_root={run_root}",
        f"total={len(results)}",
        f"exported={len(ok)}",
        f"failed={len(failed)}",
        "",
        "exports:",
    ]
    for r in results:
        lines.append(f"{r.get('func')}\t{r.get('status')}\t{r.get('export_dir')}\t{r.get('message')}")
    (run_root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_one(
    source_cpp: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    run_root: Path,
    index: int,
    total: int,
) -> dict[str, object]:
    start = time.time()
    func = source_cpp.stem
    testcase = f"{func}_msprof"
    case_root = run_root / "cases"
    case_dir = case_root / "ptoas" / testcase
    build_dir = run_root / "builds" / func
    func_root = run_root / "funcs" / func
    collect_dir = func_root / "collect"
    export_dir = func_root / "export"
    logs_dir = run_root / "logs" / func
    for p in (build_dir, func_root, collect_dir, export_dir, logs_dir):
        private_dir(p)

    result: dict[str, object] = {
        "func": func,
        "status": "started",
        "source_cpp": str(source_cpp),
        "symbol": "",
        "demangled": "",
        "app": "",
        "kernel_lib": "",
        "case_dir": str(case_dir),
        "build_dir": str(build_dir),
        "collect_dir": str(collect_dir),
        "export_dir": str(export_dir),
        "export_src": "",
        "artifact_count": 0,
        "trace_json": "",
        "visualize_data_bin": "",
        "core_trace_count": 0,
        "instr_csv_count": 0,
        "duration_sec": "",
        "message": "",
    }

    def finish(status: str, message: str) -> dict[str, object]:
        result["status"] = status
        result["message"] = message
        result["duration_sec"] = f"{time.time() - start:.2f}"
        return result

    try:
        log(f"[{index:02d}/{total:02d}] generate {func}")
        gen_cmd = [
            sys.executable,
            str(args.generate_testcase),
            "--input",
            str(source_cpp),
            "--testcase",
            testcase,
            "--output-root",
            str(case_root),
            "--run-mode",
            "sim",
            "--soc-version",
            args.soc_version,
        ]
        if args.aicore_arch:
            gen_cmd += ["--aicore-arch", args.aicore_arch]
        run_cmd(gen_cmd, env=env, log_path=logs_dir / "generate.log", timeout=args.step_timeout)

        log(f"[{index:02d}/{total:02d}] build {func}")
        cmake_cmd = [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            str(case_dir),
            "-B",
            str(build_dir),
            f"-DSOC_VERSION={args.soc_version}",
            f"-DPTO_ISA_ROOT={args.pto_isa_root}",
        ]
        run_cmd(cmake_cmd, env=env, log_path=logs_dir / "cmake.log", timeout=args.step_timeout)
        run_cmd(
            ["cmake", "--build", str(build_dir), "--target", f"{testcase}_sim"],
            env=env,
            log_path=logs_dir / "build.log",
            timeout=args.build_timeout,
        )

        app = build_dir / f"{testcase}_sim"
        kernel_lib = build_dir / f"lib{testcase}_kernel.so"
        if not app.is_file():
            raise StepError(f"missing app: {app}")
        if not kernel_lib.is_file():
            raise StepError(f"missing kernel lib: {kernel_lib}")
        result["app"] = str(app)
        result["kernel_lib"] = str(kernel_lib)

        names = [func] + [n for n in read_first_kernel_names(source_cpp) if n != func]
        symbol, demangled = resolve_symbol(kernel_lib, names)
        result["symbol"] = symbol
        result["demangled"] = demangled

        log(f"[{index:02d}/{total:02d}] golden {func}")
        run_golden(case_dir, env, logs_dir / "golden.log", args.step_timeout)

        sim_env = env.copy()
        sim_env["LD_LIBRARY_PATH"] = make_ld_library_path(build_dir, sim_env, args.soc_version)

        log(f"[{index:02d}/{total:02d}] collect {func}")
        collect_cmd = [
            "msprof",
            "op",
            "simulator",
            f"--application={app}",
            f"--kernel-name={symbol}",
            f"--launch-count={args.launch_count}",
            f"--soc-version={args.soc_version}",
            f"--timeout={args.msprof_timeout}",
            f"--output={collect_dir / 'out'}",
        ]
        cp = run_cmd(
            collect_cmd,
            cwd=case_dir,
            env=sim_env,
            log_path=collect_dir / "collect.log",
            timeout=args.msprof_timeout + 120,
            check=False,
        )
        if cp.returncode != 0 or SUCCESS_TEXT not in cp.stdout:
            tail = cp.stdout[-800:].replace("\n", " ")
            return finish("collect_failed", f"rc={cp.returncode}; tail={tail}")

        # Newer msprof op-simulator emits the final Insight traces during the
        # `collect` run itself; older versions need a separate `--export` pass.
        # Handle both: if `collect` already produced the artifacts, use them
        # and skip the (then redundant) export pass.
        collect_out = collect_dir / "out"
        artifacts = collect_artifacts(collect_out)
        if artifacts["trace_json"] and artifacts["visualize_data_bin"]:
            result.update(artifacts)
            result["export_src"] = str(collect_out)
            # Newer msprof writes the final traces during `collect`, so the real
            # artifacts live under collect_out — point export_dir there too (the
            # separate `export/` dir stays empty in this path).
            result["export_dir"] = str(collect_out)
            warn = detect_degenerate_trace(collect_out)
            log(
                f"[{index:02d}/{total:02d}] {'WARN' if warn else 'OK'} {func}: "
                f"artifacts={artifacts['artifact_count']} core_traces={artifacts['core_trace_count']} "
                f"instr_csv={artifacts['instr_csv_count']} (from collect)" + (f" -- {warn}" if warn else "")
            )
            return finish("exported", warn or "collect produced traces directly")

        export_src = find_export_src(collect_out)
        if export_src is None:
            return finish("export_src_missing", "no dump/tmp_dump under collect OPPROF dir")
        result["export_src"] = str(export_src)

        log(f"[{index:02d}/{total:02d}] export {func}")
        export_cmd = ["msprof", "op", "simulator", f"--export={export_src}", f"--output={export_dir}"]
        cp2 = run_cmd(
            export_cmd,
            cwd=case_dir,
            env=sim_env,
            log_path=export_dir / "export.log",
            timeout=args.msprof_timeout + 120,
            check=False,
        )
        artifacts = collect_artifacts(export_dir)
        result.update(artifacts)
        if cp2.returncode == 0 and artifacts["trace_json"] and artifacts["visualize_data_bin"]:
            warn = detect_degenerate_trace(export_dir)
            log(
                f"[{index:02d}/{total:02d}] {'WARN' if warn else 'OK'} {func}: "
                f"artifacts={artifacts['artifact_count']} core_traces={artifacts['core_trace_count']} "
                f"instr_csv={artifacts['instr_csv_count']}" + (f" -- {warn}" if warn else "")
            )
            return finish("exported", warn or "ok")
        tail = cp2.stdout[-800:].replace("\n", " ")
        return finish(
            "export_failed", f"rc={cp2.returncode}; artifacts={artifacts['artifact_count']}; tail={tail}"
        )
    except Exception as exc:
        return finish("failed", repr(exc))


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run a PyPTO case or use an existing build_output dir, then export "
            "msprof op-simulator Insight traces for all PTOAS funcs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_argument_group("input")
    input_group.add_argument("--build-dir", help="Existing build_output/<case> directory to export")
    input_group.add_argument(
        "--case", help="Python case script to run before exporting; args after -- are passed to it"
    )
    input_group.add_argument("--run-cmd", help="Arbitrary shell command to run before exporting")
    input_group.add_argument(
        "--task-submit", action="store_true", help="Wrap --case/--run-cmd with task-submit --run"
    )
    input_group.add_argument("--task-device", default="auto", help="task-submit --device value")
    input_group.add_argument(
        "--run-env", action="append", default=[], help="KEY=VALUE env prefix for --case; can be repeated"
    )
    input_group.add_argument(
        "--build-output-root",
        default=str(REPO_ROOT / "build_output"),
        help="Root used to locate the latest build after running a case",
    )
    input_group.add_argument("--ptoas-dir", help="Explicit directory containing source PTOAS .cpp files")
    input_group.add_argument(
        "--source-glob", help="Explicit glob for PTOAS .cpp sources; overrides default discovery"
    )
    input_group.add_argument(
        "--func", action="append", default=[], help="Only export this func/file stem; can be repeated"
    )
    input_group.add_argument(
        "--list-funcs",
        action="store_true",
        help="List discovered func names and exit before generating traces",
    )

    tool_group = parser.add_argument_group("toolchain")
    tool_group.add_argument(
        "--target",
        choices=sorted(TARGET_PROFILES),
        default="a2a3",
        help="NPU target family: sets the compile arch (dav-c220 / dav-c310) and "
        "constrains camodel-SoC auto-selection. Override with --aicore-arch / --soc-version.",
    )
    tool_group.add_argument(
        "--cann-set-env",
        default=None,
        help="CANN set_env.sh for cmake/msprof; auto-discovered when omitted",
    )
    tool_group.add_argument(
        "--ptoas-root",
        default=None,
        help="Optional PTOAS source checkout (uses its full validation-harness generator); "
        "when omitted, the bundled .pto-driven gen_profiling_case.py is used",
    )
    tool_group.add_argument(
        "--pto-isa-root", default=default_pto_isa_root(), help="pto-isa root passed to CMake"
    )
    tool_group.add_argument(
        "--soc-version",
        default=None,
        help="camodel SoC version; auto-selected within the --target family when omitted",
    )
    tool_group.add_argument(
        "--aicore-arch",
        default=os.environ.get("AICORE_ARCH"),
        help="AICore arch for generate_testcase.py; defaults from --target "
        "(a2a3 -> dav-c220, a5 -> dav-c310)",
    )

    out_group = parser.add_argument_group("output")
    out_group.add_argument(
        "--output-root", help="Directory for this export run; default is under the selected build dir"
    )
    out_group.add_argument(
        "--name",
        default="kernel_insight_all_funcs",
        help="Run directory prefix under --output-root/build dir",
    )
    out_group.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue exporting remaining funcs after one failure",
    )

    prof_group = parser.add_argument_group("profiling")
    prof_group.add_argument("--launch-count", type=int, default=1, help="msprof --launch-count")
    prof_group.add_argument(
        "--msprof-timeout", type=int, default=180, help="msprof simulator timeout seconds"
    )
    prof_group.add_argument(
        "--step-timeout", type=int, default=300, help="timeout for testcase generation/cmake/golden"
    )
    prof_group.add_argument("--build-timeout", type=int, default=600, help="timeout for cmake build per func")

    args, case_args = parser.parse_known_args(argv)
    if case_args and case_args[0] == "--":
        case_args = case_args[1:]
    args.ptoas_root = repo_path(args.ptoas_root) if args.ptoas_root else None
    args.pto_isa_root = repo_path(args.pto_isa_root)
    args.generate_testcase = resolve_generate_testcase(args.ptoas_root)
    return args, case_args


def assert_bisheng_supports_tl(env: dict[str, str], aicore_arch: str) -> None:
    """Fail fast if the CANN bisheng on PATH ignores ``--cce-aicore-enable-tl``.

    The generated CMake compiles every kernel with ``--cce-aicore-enable-tl``
    (the Tile-Language extensions the pto-isa headers require). A non-TL CANN
    (commonly ``ascend-toolkit/latest``, e.g. 8.3.RC1) silently drops the flag and
    the per-kernel build then dies deep inside ``pto/common/memory.hpp`` with
    ``unknown type name '__biasbuf__'`` / ``use of undeclared identifier
    'aicore'``. Probe once, up front, so the root cause is named instead of
    deferred to a cryptic C++ error after a full generate+cmake cycle.
    """
    bisheng = shutil.which("bisheng", path=env.get("PATH"))
    if not bisheng:
        return  # cannot probe here; the build step will surface any issue
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "tl_probe.cpp"
            # __biasbuf__ is a TL-only address-space qualifier: a TL-capable
            # bisheng compiles this; a non-TL one errors `unknown type name
            # '__biasbuf__'` (the exact symptom seen deep in pto-isa headers).
            src.write_text("__biasbuf__ int *probe_ptr;\n")
            cp = subprocess.run(
                [
                    bisheng,
                    "-xcce",
                    "-c",
                    str(src),
                    "-o",
                    str(Path(tmp) / "o.o"),
                    "--cce-aicore-enable-tl",
                    f"--cce-aicore-arch={aicore_arch}",
                    "-std=c++17",
                ],
                cwd=tmp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            out = cp.stdout or ""
    except OSError:
        return  # probe could not run; let the build surface any issue
    if "__biasbuf__" in out and "unknown type" in out:
        raise StepError(
            f"CANN bisheng at {bisheng} does NOT support --cce-aicore-enable-tl "
            "(Tile-Language). This is a non-TL CANN (commonly ascend-toolkit/latest, "
            "e.g. 8.3.RC1). The per-kernel build would fail with '__biasbuf__' / "
            "undeclared 'aicore'. Re-run with a TL-capable --cann-set-env, e.g. "
            "--cann-set-env /path/to/cann-8.5.x/set_env.sh (the toolchain used for the "
            "device build, often the one ASCEND_HOME_PATH already points to)."
        )


def validate_toolchain(args: argparse.Namespace) -> dict[str, str]:
    """Validate the PTOAS/CANN toolchain and return the sourced environment."""
    if args.cann_set_env is None:
        args.cann_set_env = discover_cann_set_env()
        if args.cann_set_env is None:
            raise StepError("CANN set_env.sh not found; pass --cann-set-env explicitly")
    if not args.generate_testcase.is_file():
        raise StepError(f"generate_testcase.py not found: {args.generate_testcase}")
    if not args.pto_isa_root.exists():
        raise StepError(f"--pto-isa-root not found: {args.pto_isa_root}")
    env = source_env(args.cann_set_env, os.environ.copy())
    if not shutil.which("msprof", path=env.get("PATH")):
        raise StepError("msprof not found after sourcing CANN environment")
    return env


def resolve_build_dir(
    args: argparse.Namespace,
    build_output_root: Path,
    run: list[str] | None,
    env: dict[str, str],
) -> Path:
    """Return the case build directory, running the case command if needed."""
    if args.build_dir:
        return repo_path(args.build_dir)
    if not run:
        raise StepError("provide --build-dir, --case, or --run-cmd")
    before = build_output_dirs(build_output_root)
    start = time.time()
    log("running case command")
    run_log_root = build_output_root / f"kernel_insight_case_run_{timestamp()}"
    private_dir(run_log_root)
    run_cmd(run, cwd=REPO_ROOT, env=env, log_path=run_log_root / "case_run.log", timeout=None, check=True)
    build_dir = select_latest_build(build_output_root, before, start)
    log(f"selected build dir: {build_dir}")
    return build_dir


def main(argv: list[str] | None = None) -> int:
    args, case_args = parse_args(argv or sys.argv[1:])
    build_output_root = repo_path(args.build_output_root)
    run = build_run_command(args, case_args)

    # Func discovery is filesystem-only: `--list-funcs` against an existing
    # `--build-dir` needs neither the CANN toolchain nor the PTOAS sources.
    # --target supplies the compile arch unless --aicore-arch was given explicitly.
    if not args.aicore_arch:
        args.aicore_arch = TARGET_PROFILES[args.target]["aicore_arch"]

    needs_toolchain = not (args.list_funcs and args.build_dir)
    if needs_toolchain:
        env = validate_toolchain(args)
        assert_bisheng_supports_tl(env, str(args.aicore_arch))
        args.soc_version = select_soc_version(args.target, discover_camodel_socs(env), args.soc_version)
    else:
        env = os.environ.copy()
    build_dir = resolve_build_dir(args, build_output_root, run, env)

    if not build_dir.is_dir():
        raise StepError(f"build dir not found: {build_dir}")

    sources = default_ptoas_sources(
        build_dir,
        repo_path(args.ptoas_dir) if args.ptoas_dir else None,
        args.source_glob,
    )
    if args.func:
        wanted = set(args.func)
        sources = [p for p in sources if p.stem in wanted]
    if not sources:
        raise StepError(f"no PTOAS .cpp sources found for build dir: {build_dir}")
    if args.list_funcs:
        for source in sources:
            print(f"{source.stem}\t{source}")
        return 0

    if args.output_root:
        base_output = repo_path(args.output_root)
    else:
        base_output = build_dir
    run_root = base_output / f"{args.name}_{timestamp()}"
    private_dir(run_root)
    pointer_file = build_dir / "latest_all_funcs_kernel_insight_export_root.txt"
    pointer_file.write_text(str(run_root) + "\n", encoding="utf-8")

    metadata = [
        f"repo_root={REPO_ROOT}",
        f"build_dir={build_dir}",
        f"source_count={len(sources)}",
        f"cann_set_env={repo_path(args.cann_set_env)}",
        f"soc_version={args.soc_version}",
        f"generate_testcase={args.generate_testcase}",
        f"pto_isa_root={args.pto_isa_root}",
        "sources:",
        *[str(p) for p in sources],
    ]
    (run_root / "README.txt").write_text("\n".join(metadata) + "\n", encoding="utf-8")

    fieldnames = [
        "func",
        "status",
        "source_cpp",
        "symbol",
        "demangled",
        "app",
        "kernel_lib",
        "case_dir",
        "build_dir",
        "collect_dir",
        "export_dir",
        "export_src",
        "artifact_count",
        "trace_json",
        "visualize_data_bin",
        "core_trace_count",
        "instr_csv_count",
        "duration_sec",
        "message",
    ]
    results: list[dict[str, object]] = []
    for idx, source_cpp in enumerate(sources, 1):
        result = export_one(source_cpp, args, env, run_root, idx, len(sources))
        results.append(result)
        write_outputs(run_root, results, fieldnames)
        if result.get("status") != "exported" and not args.keep_going:
            break

    failed = [r for r in results if r.get("status") != "exported"]
    log(f"RUN_ROOT {run_root}")
    log(f"EXPORTED {len(results) - len(failed)}/{len(results)}")
    if failed:
        log("FAILED_FUNCS " + ",".join(f"{r.get('func')}:{r.get('status')}" for r in failed))
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
