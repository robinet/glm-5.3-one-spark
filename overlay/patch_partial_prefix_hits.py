#!/usr/bin/env python3
"""Opt-in fine-grained (sub-block) prefix-cache hits for GLM-5.3 (ONE_SPARK_PARTIAL_APC=1).

``HybridKVCacheCoordinator`` disables fine-grained hits whenever any cache
manager lacks ``supports_fine_grained_hash_lookup`` and its block size differs
from ``hash_block_size``. For GLM5Next that is only ``KpoolTailManager``, so
every prefix hit is rounded down to a whole 7168-token MLA/Mamba block and up
to 7167 tokens of a cached conversation are re-prefilled each turn.

The tail never takes part in prefix lookup: it is per-request scratch that
holds only the in-progress (incomplete) indexer pool, and it opts out of prefix
caching. A hit at ``H`` needs no tail state when ``H`` ends on a complete pool
(``H % index_kpool == 0``) -- the same situation as today's 7168-aligned hits.
Fine-grained hits land on ``hash_block_size`` boundaries, and the tail block
size is a multiple of ``index_kpool``, so this exempts the tail from the gate
only when ``hash_block_size % tail.block_size == 0``. MLA, compressed-indexer
and Mamba partial-block state keep using vLLM's existing copy-on-write
(``copy_kv_cache_blocks_inplace`` copies whole pages of every KV storage).

Requires the mamba prefix-hit seed fix (vllm-project/vllm#55600, applied by
``scripts/serve-one-spark.sh`` in PR #4): without it any prefix hit can restore
the wrong KDA state, and partial hits would make that far more frequent. This
script refuses to enable itself unless that fix is present.

Run at container start (after the seed fix), not at image build, so the switch
stays opt-in. Fail-closed, idempotent, preflights before writing, atomic replace.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

SITE = Path(os.environ.get("GLM53_VLLM_SITE", "/usr/local/lib/python3.12/dist-packages/vllm"))
COORD = SITE / "v1/core/kv_cache_coordinator.py"
MAMBA_HYBRID = SITE / "v1/worker/gpu/model_states/mamba_hybrid.py"

MARK = "[glm53-partial-apc]"
GATE = (
    "                if not manager.supports_fine_grained_hash_lookup\n"
    "                and manager.block_size != hash_block_size\n"
)
GATE_NEW = GATE + (
    f"                and not (  # {MARK} tail never looks up; pool-aligned hits need no tail state\n"
    "                    type(manager).__name__ == \"KpoolTailManager\"\n"
    "                    and hash_block_size % manager.block_size == 0\n"
    "                )\n"
)
SEED_FIXED = "(new_req_data.num_computed_tokens - 1) // self.cache_config.mamba_block_size"
SEED_UNFIXED = "(new_req_data.num_computed_tokens - 1) // self.cache_config.block_size"


def prepare(source: str) -> tuple[str, str]:
    """Idempotent, fail-closed. Returns (text, action)."""
    if MARK in source:
        if source.count(GATE_NEW) != 1:
            raise ValueError("partial/inconsistent partial-APC patch -- refusing to touch a half-patched file")
        return source, "already present"
    n = source.count(GATE)
    if n != 1:
        raise ValueError(f"pinned anchor 'fine-grained gate' drifted (found {n}, expected 1)")
    out = source.replace(GATE, GATE_NEW, 1)
    if out.count(GATE_NEW) != 1:
        raise ValueError("partial-APC post-patch verification failed")
    return out, "patched"


def check_seed_fix(text: str) -> None:
    """Refuse unless the vllm#55600 mamba seed fix is applied."""
    if SEED_UNFIXED in text or SEED_FIXED not in text:
        raise ValueError("mamba prefix-hit seed fix (vllm#55600) is not applied; partial hits would restore "
                         "the wrong KDA state. Apply it first (scripts/serve-one-spark.sh, PR #4).")


def replace_file(target: Path, text: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-partial-apc.tmp")
    try:
        tmp.write_text(text)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{target.stem}*.pyc"):
            pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    preflight_only = "--preflight" in argv[1:]
    for target in (COORD, MAMBA_HYBRID):
        if not target.is_file():
            raise SystemExit(f"missing {target}")
    source = COORD.read_text()
    try:
        check_seed_fix(MAMBA_HYBRID.read_text())
        text, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"partial prefix-cache hits: preflight failed: {exc}") from exc
    compile(text, str(COORD), "exec")
    if not preflight_only and text != source:
        replace_file(COORD, text)
        clear_pyc(COORD)
    print(f"{COORD.name}: fine-grained prefix-cache hits (KpoolTail exemption) {action}"
          + (" [preflight only]" if preflight_only else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
