#!/usr/bin/env python3
"""Thin wrapper: same as ``python -m ssr.retrieval.eval_mteb --task index ...``.

End-to-end MTEB index build: stream corpus.jsonl → GPU encode → CPU index → cache on disk.

Example:
  .venv/bin/python scripts/build_mteb_index_e2e.py \\
    --model-path output/.../gpu4-topk-32/final \\
    --dataset nfcorpus \\
    --index-cache-dir data/cache/mteb_index_e2e/nfcorpus \\
    --encode-device cuda:0
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ssr import _bootstrap  # noqa: F401


def main(argv: list[str] | None = None) -> None:
    raw = list(argv) if argv is not None else sys.argv[1:]
    if "--task" not in raw:
        raw = ["--task", "index", *raw]
    from ssr.retrieval.eval_mteb import main as eval_main

    eval_main(raw)


if __name__ == "__main__":
    main()
