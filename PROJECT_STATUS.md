# PROJECT_STATUS — warranty-claim-recovery

**Last updated:** 2026-09-24

---

## Current milestone

**M0 — the contract is committed and no source file exists.**

`DECISIONS.md` ADR-001 fixes three claims, thirteen kill conditions with their thresholds, four
retrieval baselines and the hold-out rule. `tests/test_kill_criteria.py` grades them and imports
nothing from the package. `tests/test_predeclaration.py` parses the kill test and asserts its
thresholds are module-level literals, that none has been lowered, that the absolute zeros are zero,
that it cannot skip itself, and that every criterion states the population it was graded over.

This ordering is the whole point and it is checkable in `git log`: the commit carrying the contract
contains **no implementation and no score**.

## Authoritative obligations this project alone carries

Recovered from `PORTFOLIO_BLUEPRINT.md` §8 and `SKILL_MATRIX.md`; ADR-001 §1 has the table.

- LangGraph agent state machine, durable checkpointing, crash resumability — **sole home**
- HITL breakpoints on write mutations — mandated coverage names only this project
- RAG through citation with measured retrieval evaluation — **no slack**, project 3 shipped without it
- pgvector — **no slack**, 7 and 8 carry it alone
- Redis and Redis-backed queues, idempotency keys, failure-injection suite
- Audit-event contract as an independent implementation, policy enforcement
- Groundedness scoring, retrieval-vs-generation attribution, eval as a build-failing CI gate

## Known divergences, recorded up front

| blueprint | built | why |
|---|---|---|
| DigitalOcean App Platform | Render Free + Neon Free + Upstash Free | no free tier for a web service; no payment authorisation |
| Next.js console | server-rendered HTML unless a build step earns itself | same decision as projects 5, 6 and 7 |
| the model proposes the correction | extractive composer ships; the abstractive arm **raises** | no model credential exists here, and a stub would enter the evaluation as a number attributed to a model that was never called |

## Verified

- `tests/test_predeclaration.py` passes against a repository with no `src/` at all, which is the
  only moment it can prove it does not depend on the system under test.

## Blockers

None requiring the owner. A model API key would enable the abstractive arm; the project is designed
to ship, measure and deploy without one.

## Next

M1 data model and migrations, M2 the deterministic core, then the corpus and the frozen hold-out —
in that order, so the hold-out is frozen before anything can be scored against it.
