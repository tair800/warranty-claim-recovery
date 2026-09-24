# CLAUDE.md — warranty-claim-recovery

The operating contract for this repository. Read it before touching anything here.

`DECISIONS.md` is authoritative and this file is subordinate to it. **ADR-001 is the predeclared
contract**: three claims, thirteen kill conditions with fixed thresholds, four retrieval baselines,
and the hold-out rule, all committed before any source file existed. Where this file and
`DECISIONS.md` disagree, `DECISIONS.md` wins and this file is the thing to fix.

---

## 1. Purpose

A distributor submits warranty claims to manufacturers and gets them back rejected for missing or
inconsistent evidence. Corrections must be filed before the manufacturer's claim window closes or the
money is written off, and cases run for days waiting on technician and customer replies.

So the expensive failures are not slow answers. They are: a resubmission that should never have gone
out, the same resubmission going out twice, a case lost because a worker restarted mid-flight, and a
claim written off because nobody noticed the window closing. Every one of those is a kill condition.

**This project is the portfolio's sole home for LangGraph, durable checkpointing and crash
resumability.** It also carries the RAG-through-citation mandate and the pgvector mandate with no
slack, because project 3 shipped without a retrieval layer. See ADR-001 §1.

---

## 2. Architecture

### 2.1 The case machine

One LangGraph graph with a PostgreSQL checkpointer. Explicit nodes, explicit edges, no dynamic
routing by model output. The order is the argument:

```
intake            normalise the rejection, resolve the program and policy version
requirements      the rejection code's requirement matrix, deterministic
deadline          claim-window arithmetic, deterministic
eligibility       warranty period, part and serial, prior recovery, deterministic
retrieval         the governing clause, from pgvector, with citation
compose           the correction, from cited spans
gate              RECOVERABLE / PARTIALLY_RECOVERABLE / NOT_RECOVERABLE / REVIEW
approval          interrupt — a person decides, and the graph stops here until they do
submit            idempotent, at most one effect per recovery identity
```

**Every node is resumable and every tool invocation is recorded before it runs and completed after.**
That pair is what makes "no tool re-execution after a kill" checkable rather than hoped for.

`approval` is a real LangGraph interrupt, not a flag. The graph **stops**, the state is durable, and
a different process on a different day resumes it.

### 2.2 The deterministic core

Owns, and no model may touch: claimed amount, eligible amount, excluded amount, deductible, cap,
recoverable amount, warranty period, part and serial eligibility, supplier responsibility, claim
window, prior recovery, duplicate detection, and the gate outcome.

**Money is `Decimal`.** No float touches a monetary value anywhere, including in tests and fixtures.
Currency is explicit on every amount and a cross-currency comparison raises rather than converting.
Rounding is declared once, in one module, and is applied at exactly one point.

### 2.3 Retrieval

Warranty policy clauses and service bulletins, chunked, embedded, stored in a `vector` column,
retrieved with the pgvector distance operator and filtered by manufacturer and policy version
**before** ranking. A citation is a verbatim span plus its offsets into the document version it names.

### 2.4 Redis

Load-bearing, not decorative: the case work queue, its **leases**, and submission idempotency state.
A worker leases a case; if it dies the lease expires and another worker resumes from the checkpoint.
With Redis gone the queue **fails closed** — kill condition L.

---

## 3. Non-negotiable rules

1. **`tests/test_kill_criteria.py` and `tests/test_predeclaration.py` are PREDECLARED.** Thresholds
   may be raised, never lowered. Neither file may be edited to make a build pass. If a measurement
   misses, fix the system or record the failure in `DECISIONS.md`.
2. **The hold-out is frozen before it is scored, split by warranty program, and never re-drawn to
   improve a number.** Nothing may be tuned against it after scoring.
3. **No model may decide money, eligibility, a deadline or the gate outcome.** The gate is computed
   before any text is composed. `RECOVERABLE` is reachable from exactly one place.
4. **Nothing leaves the system without a recorded human approval for the same case version**, and one
   recovery identity produces at most one submission effect.
5. **Never claim a number that was not measured.** Every figure in the README, an artifact or the
   console comes from a committed code path that reproduces it.
6. **Guards must fail from BEHAVIOUR.** A breach is planted by replacing a function the running
   system calls, and the detector observes the behaviour change. A grep over source text is not a
   guard.
7. **`Decimal` only for money.** A float in a monetary path is a defect, not a style preference.
8. **The corpus is synthetic and is described as synthetic everywhere**, including in the body of
   every generated file and on every screen. It is never presented as a manufacturer publication.
9. **No secrets in the repository.** `.env` is gitignored; `.env.example` carries placeholders.
   `WCR_APPROVER_TOKEN` and `WCR_LLM_API_KEY` have no defaults — both fail closed.
10. **No multi-agent architecture, no Kubernetes, no Kafka, no Temporal.** All recorded portfolio-wide
    omissions; the master spec forbids complexity that only decorates a README.

---

## 4. Conventions

- Typed Python 3.12, `from __future__ import annotations`, `src/` layout, `uv` with a committed lock.
- Pydantic v2 models frozen with `extra="forbid"`. `extra="forbid"` does real work: a mistyped field
  silently becoming an ignored attribute is how an eligibility check comes to check nothing.
- ruff at 100 columns, `ruff format --check` part of lint. mypy `--strict` over `src`, not advisory.
- **Docstrings argue WHY, at length, and name the failure they prevent.** British English, plain, no
  marketing adjectives. A rejected alternative is recorded as rejected, with its reason.
- Constants that appear in an artifact are imported from the code that implements them, so the claim
  and the implementation are the same string.
- Deterministic output: no wall clock in a benchmark, no `date.today()` default in a model, no set
  iteration deciding an order.

---

## 5. Key commands

`make help` lists them. The chain a reviewer would run:

```
make setup       uv sync
make db          PostgreSQL+pgvector on 127.0.0.1:15441, Redis on 127.0.0.1:16380
make migrate     alembic upgrade head
make corpus      the synthetic corpus, from its committed seed
make index       embed and load the clause corpus
make artifacts   run everything and write the evidence
make breaches    plant a defect into every guarantee and check each is caught
make test        the engineering suite
make release-gate the predeclared kill test, reported
make console     the Warranty Recovery Lab
make evidence    the full chain, and what CI runs
```

---

## 6. Deployment shape

Render Free (Docker, Frankfurt) + Neon Free PostgreSQL with pgvector + Upstash Free Redis.
The blueprint named DigitalOcean App Platform; it has no free tier for a web service and no payment
authorisation exists. ADR-001 §3 records the divergence.

**Measure the encoder's resident memory before choosing the topology.** Project 7 shipped a
multilingual model with a 250,000-token vocabulary onto a 512MB instance and the container was killed
on every query. This corpus is English-only, so a small English encoder is the right default — but
the number is measured and recorded before it is relied on, not assumed.

---

## 7. Current integrations

- **PostgreSQL 16 + pgvector** — the store, the vector index, and the LangGraph checkpointer.
- **Redis** — work queue, leases, submission idempotency. Load-bearing.
- **fastembed, local ONNX** — English-only encoder, no network call at query time.
- **No model provider.** `WCR_LLM_API_KEY` has no default and the abstractive arm **raises**.
