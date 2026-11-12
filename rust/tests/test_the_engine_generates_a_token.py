"""The engine's token path, under a gate.

`decode_one_token` is the goal's item (3) in one function: final norm, head, argmax, token id. The
weights are 288 GB and the model build is the blocker, so this test pins the MECHANISM with a model
whose answer is known by construction -- a zero-layer model with a permutation head. If the tail of
the decode path breaks, this fails before anyone re-measures it against a package.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "rust" / "ds4-rs-ascend"


def test_the_engine_produces_the_token_its_head_selects():
    out = subprocess.run(
        ["cargo", "run", "--release", "--offline", "--example", "w8a8_token"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    text = out.stdout + out.stderr
    assert out.returncode == 0, f"the example failed:\n{text[-2000:]}"
    assert "PASS: the engine produced the token its own head selects" in text, text[-2000:]
    assert "token from the engine:" in text, text[-2000:]


def _run(example: str, needle: str) -> None:
    out = subprocess.run(
        ["cargo", "run", "--release", "--offline", "--example", example],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    text = out.stdout + out.stderr
    assert out.returncode == 0, f"{example} failed:\n{text[-2000:]}"
    assert needle in text, text[-2000:]


def test_a_one_layer_model_runs_attention_and_moe_through_the_decode_path():
    """Zero weights make the layer a PURE RESIDUAL, so the token must not move; a single large
    cross-term must move it. The pair is what separates "the sub-blocks ran" from "the sub-blocks
    did nothing", and the second half is the one a silent no-op would fail."""
    _run("w8a8_layer_token", "PASS: a zero-weight layer is a pure residual")
