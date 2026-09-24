"""Measure the encoder's resident memory before a deployment topology is chosen.

Project 7 shipped a multilingual model with a 250,000-token vocabulary onto a 512MB instance. The
loaded session measured 671MB and the container was killed on every query; the fix cost a day and
constrained the deployment permanently. The number was knowable in advance and nobody measured it.

So this measures it. It loads the encoder, embeds a realistic batch, and reports the peak resident
set of the whole process against the free tier's ceiling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FREE_TIER_MB = 512


def rss_mb() -> float:
    import psutil  # noqa: PLC0415

    return float(psutil.Process(os.getpid()).memory_info().rss) / 1_048_576


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args(argv)

    baseline = rss_mb()
    from fastembed import TextEmbedding  # noqa: PLC0415

    imported = rss_mb()

    model = TextEmbedding(model_name=args.model, cache_dir=str(REPO_ROOT / ".fastembed_cache"))
    loaded = rss_mb()

    texts = [
        f"Clause {i}: the warranty period for a replaced assembly runs from the repair invoice "
        f"date and not from the original sale, and a claim filed after ninety days is refused."
        for i in range(args.batch)
    ]
    vectors = list(model.embed(texts))
    embedded = rss_mb()

    payload = {
        "model": args.model,
        "dimensions": len(vectors[0]),
        "batch": args.batch,
        "rss_mb": {
            "baseline": round(baseline, 1),
            "after_import": round(imported, 1),
            "after_load": round(loaded, 1),
            "after_embedding": round(embedded, 1),
        },
        "peak_mb": round(embedded, 1),
        "free_tier_mb": FREE_TIER_MB,
        "fits_free_tier_with_headroom": embedded < FREE_TIER_MB * 0.6,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
