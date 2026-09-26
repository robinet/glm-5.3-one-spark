#!/usr/bin/env python3
"""Regression tests for the opt-in partial prefix-cache-hit overlay (ONE_SPARK_PARTIAL_APC).

CPU-only: pinned-fixture patching, idempotence, fail-closed refusal of a half-patched or drifted file,
refusal without the vllm#55600 mamba seed fix, and that the patched gate exempts only a pool-aligned
KpoolTailManager. Inside the image it also preflights the installed coordinator anchor.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (HERE / "patch_partial_prefix_hits.py", ROOT / "overlay" / "patch_partial_prefix_hits.py")
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
import patch_partial_prefix_hits as pp  # noqa: E402

FIXTURE = (
    "        if self.enable_partial_hash_hits:\n"
    "            unsupported_partial_hit_managers = {\n"
    "                type(manager).__name__\n"
    "                for manager in self.single_type_managers\n"
    + pp.GATE
    + "            }\n"
)


def _expect(fn, needle):
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), exc
    else:
        raise AssertionError(f"expected refusal containing {needle!r}")


def test_fixture_roundtrip():
    patched, action = pp.prepare(FIXTURE)
    assert action == "patched" and pp.MARK in patched
    again, action = pp.prepare(patched)
    assert action == "already present" and again == patched
    compile("class C:\n    def f(self, hash_block_size):\n" + patched, "fixture", "exec")


def test_half_patched_is_refused():
    patched, _ = pp.prepare(FIXTURE)
    _expect(lambda: pp.prepare(patched.replace("hash_block_size % manager.block_size == 0", "True")), "half-patched")


def test_drifted_anchor_is_refused():
    _expect(lambda: pp.prepare(FIXTURE.replace("manager.block_size != hash_block_size", "False")), "drifted")


def test_requires_mamba_seed_fix():
    _expect(lambda: pp.check_seed_fix(f"idx = {pp.SEED_UNFIXED}\n"), "vllm#55600")
    _expect(lambda: pp.check_seed_fix("no seed line at all\n"), "vllm#55600")
    pp.check_seed_fix(f"idx = {pp.SEED_FIXED}\n")


class _Mgr:
    supports_fine_grained_hash_lookup = False

    def __init__(self, block_size):
        self.block_size = block_size


def _unsupported(managers, hash_block_size):
    """Evaluate the patched gate's set comprehension on stand-in managers."""
    patched, _ = pp.prepare(FIXTURE)
    body = patched.split("unsupported_partial_hit_managers = ", 1)[1]
    return eval(body, {"hash_block_size": hash_block_size, "self": type("S", (), {"single_type_managers": managers})})


def test_gate_exempts_only_pool_aligned_tail():
    KpoolTailManager = type("KpoolTailManager", (_Mgr,), {})
    Other = type("SlidingWindowManager", (_Mgr,), {})
    assert _unsupported([KpoolTailManager(16)], 1024) == set()  # ring 16 | 1024 -> exempt
    assert _unsupported([KpoolTailManager(4)], 64) == set()  # stock ring 4 | drafter block 64
    assert _unsupported([KpoolTailManager(16)], 1000) == {"KpoolTailManager"}  # not pool-aligned
    assert _unsupported([Other(16)], 1024) == {"SlidingWindowManager"}  # other managers still block


def test_installed_files_preflight():
    """Inside the image: the pinned coordinator anchor matches (or is already patched)."""
    if not pp.COORD.is_file():
        return  # not in the image
    _text, action = pp.prepare(pp.COORD.read_text())
    assert action in ("patched", "already present")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} partial prefix-hit tests passed")
