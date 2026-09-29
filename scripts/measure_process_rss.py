"""Measure the resident memory of the process that will actually run, not of the model inside it.

    python scripts/measure_process_rss.py

Writes `artifacts/process_memory.json`.

`artifacts/encoder_memory.json` measured the encoder alone at 285MB against a 512MB instance, and
that number was the right one to take first: project 7 shipped a model that did not fit and learned
it from a container killed on every query. But an encoder is not a deployment. The process that
serves the console also holds FastAPI and uvicorn, a SQLAlchemy engine and its psycopg connections,
the LangGraph runtime and its PostgreSQL checkpointer, the templates, and whatever each of those
allocates the first time a real request touches it. Treating the model's footprint as the
process's is the same mistake as project 7's, one layer up.

So this starts the real server as a child process, drives it through every screen with a real
claim — including the evidence screen with retrieval switched on, which is the only path that
loads the encoder — and samples the child's resident set after each step. It also loads the graph
runtime and compiles the case machine inside a second child, because the console's checkpoint view
reads it, and a measurement that left it out would be a measurement of a process that will not be
deployed.

The figure published is the peak, not the settled value. A container is killed on its peak.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "artifacts" / "process_memory.json"
FREE_TIER_MB: Final = 512
#: A process at 60% of its ceiling has room for a traffic spike and for the allocator's
#: fragmentation over a long uptime; one at 90% is waiting for the request that kills it.
SAFE_FRACTION: Final = 0.6
DEFAULT_CLAIM: Final = "kestrel-hydraulics-2019.1-c15"
HTTP_OK: Final = 200


def rss_mb(pid: int) -> float:
    """The resident set of a process **and every descendant**, in megabytes.

    The tree rather than the process, and the reason is a measurement this script got wrong the
    first time it ran. On Windows a virtual environment's `python.exe` is a launcher stub that
    starts the real interpreter as a child and waits on it, so the process `Popen` returns is the
    stub, and its resident set is about 5MB. The first run published "server peak 5.1MB" for a
    process running FastAPI, SQLAlchemy and an ONNX session — a number that would have made the
    deployment look forty times safer than it is. On Linux, where the container runs, there is no
    stub and the tree is one process; summing it is correct on both, and on Windows it over-counts
    by the stub's few megabytes, which is the safe direction for a figure whose job is to say
    whether a container will be killed.
    """
    import psutil  # noqa: PLC0415

    root = psutil.Process(pid)
    total = 0
    for process in (root, *root.children(recursive=True)):
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return float(total) / 1_048_576


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_until_up(base: str, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/livez", timeout=3) as response:  # noqa: S310
                if response.status == HTTP_OK:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"the server at {base} never answered /livez")


def fetch(base: str, path: str) -> int:
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=180) as response:  # noqa: S310
            response.read()
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def measure_server(claim: str, repeats: int) -> dict[str, Any]:
    """The console process, driven through every screen, sampled after each."""
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "WCR_READ_ONLY": "true", "PYTHONUNBUFFERED": "1"}
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "warranty_claim_recovery.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    samples: list[dict[str, Any]] = []
    try:
        wait_until_up(base)
        samples.append({"step": "started, no request served", "rss_mb": rss_mb(server.pid)})

        steps = [
            ("/", "case screen"),
            (f"/?claim={claim}", "case screen with a claim"),
            (f"/recovery?claim={claim}", "recovery calculation"),
            (f"/evidence?claim={claim}&retrieve=true", "evidence with live retrieval (encoder)"),
            ("/evaluation", "evaluation"),
            (f"/audit?claim={claim}", "audit"),
            (f"/api/case/{claim}", "case JSON"),
            ("/healthz", "readiness"),
        ]
        for _ in range(repeats):
            for path, label in steps:
                status = fetch(base, path)
                samples.append(
                    {"step": label, "path": path, "status": status, "rss_mb": rss_mb(server.pid)}
                )
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()

    peak = max(sample["rss_mb"] for sample in samples)
    return {
        "peak_mb": round(peak, 1),
        "started_mb": round(samples[0]["rss_mb"], 1),
        "samples": [{**item, "rss_mb": round(item["rss_mb"], 1)} for item in samples],
    }


_GRAPH_PROBE: Final = """
import os, sys, json, psutil
sys.path.insert(0, 'src')
proc = psutil.Process(os.getpid())
mb = lambda: proc.memory_info().rss / 1048576
out = {'baseline': mb()}
from warranty_claim_recovery.retrieval.pipeline import Retriever
from warranty_claim_recovery.retrieval.embeddings import FastEmbedEncoder
enc = FastEmbedEncoder()
enc.encode_query('the serial plate is missing from the claim')
out['with_encoder'] = mb()
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from warranty_claim_recovery.graph import machine
out['with_graph_runtime'] = mb()
print(json.dumps(out))
"""


def measure_graph_runtime() -> dict[str, Any]:
    """The encoder and the LangGraph runtime together, in one fresh process.

    Separate from the server measurement because the console does not yet load the graph at
    import, and measuring it only through a route that happens to touch it would under-report a
    deployment in which it does.
    """
    result = subprocess.run(
        [sys.executable, "-c", _GRAPH_PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    values: dict[str, float] = json.loads(result.stdout.strip().splitlines()[-1])
    return {key: round(value, 1) for key, value in values.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", default=DEFAULT_CLAIM)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    server = measure_server(args.claim, args.repeats)
    graph = measure_graph_runtime()
    # The deployed process is the server plus whatever the graph runtime adds on top of an
    # encoder-loaded interpreter. Taken as the server's peak plus the graph's increment over the
    # encoder, which over-counts shared libraries rather than under-counting them — the safe
    # direction for a number whose job is to say whether a container will be killed.
    graph_increment = max(0.0, graph["with_graph_runtime"] - graph["with_encoder"])
    projected = round(server["peak_mb"] + graph_increment, 1)

    payload = {
        "is_synthetic_corpus": True,
        "what_this_is": (
            "the resident memory of the whole console process — web server, database client, "
            "encoder and graph runtime — measured while it serves every screen, rather than the "
            "encoder's footprint alone"
        ),
        "why": (
            "encoder_memory.json measured the model at 285MB. A model is not a deployment; "
            "project 7's container was killed because nobody measured what actually runs"
        ),
        "free_tier_mb": FREE_TIER_MB,
        "safe_fraction": SAFE_FRACTION,
        "server": server,
        "graph_runtime_probe": graph,
        "graph_runtime_increment_mb": round(graph_increment, 1),
        "projected_deployed_peak_mb": projected,
        "fits_free_tier": projected < FREE_TIER_MB,
        "fits_with_headroom": projected < FREE_TIER_MB * SAFE_FRACTION,
    }
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"server peak {server['peak_mb']}MB (started {server['started_mb']}MB); graph runtime adds "
        f"{graph_increment:.1f}MB; projected deployed peak {projected}MB of {FREE_TIER_MB}MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
