# Decisions

Architecture decision records for `warranty-claim-recovery`.

**ADR-001 is the predeclared contract. It was committed before any source file existed, and the
commit that carries it contains no implementation and no score.** Its thirteen kill conditions and
their thresholds may be raised. They may never be lowered, deleted, renamed, or marked `xfail`.

---

## ADR-001 — The contract, recovered from the portfolio sources, and the gates that can kill it

**Status:** accepted · **Date:** 2026-09-24 · **Committed before any implementation.**

### 1. What the authoritative sources require

Recovered from `PORTFOLIO_BLUEPRINT.md` §8, `SKILL_MATRIX.md` and `PORTFOLIO_MASTER_SPEC.md`. Where
this project's brief and the authoritative sources disagree, the authoritative sources win; §2 below
records exactly where that happened and what was done about it.

**The blueprint's single purpose, verbatim in substance:** *a resumable case workflow that turns a
manufacturer's rejected warranty claim into a corrected, evidence-backed resubmission before the
claim window closes — and survives being killed mid-case.*

**The blueprint's headline claim, verbatim:**

> *Kill the worker mid-case and the workflow resumes at the same node with no duplicate submission
> and no lost human decision, and no resubmission leaves the system without a recorded approval —
> demonstrated by a kill-and-resume test in CI and an idempotent submission test, not described in
> prose — while correction proposals are scored for citation groundedness against a committed golden
> set of rejected-claim cases with retrieval-versus-generation attribution.*

**Obligations this project alone carries.** Each is a `●` in `SKILL_MATRIX.md` with no second home,
or a mandated-coverage row whose slack has already been spent:

| obligation | source | why it has no slack |
|---|---|---|
| LangGraph agent state machine | matrix §LLM, sole home | "If it is cut, the portfolio has no stateful-agent evidence at all." |
| Durable checkpointing and crash resumability | matrix §LLM, sole home | same row |
| HITL breakpoints on write mutations | mandated coverage: "One LangGraph stateful agent with durable checkpointing + HITL breakpoints — **Met — 8**" | the row names only this project |
| **RAG end-to-end through citation with measured retrieval evaluation** | mandated coverage | "**NOT MET as built — 2 of 3.** Project 3 shipped without any retrieval layer, so 7 and 8 now carry the requirement alone and **have no slack**." |
| **pgvector** | mandated coverage | "**7 and 8 now carry pgvector alone.**" |
| Embeddings | matrix §Retrieval | 3 absent, 5 absent; 7 and 8 |
| Redis, and queues on Redis | matrix §Backend | `●` for 8; project 5 dropped its cell |
| Idempotency keys | matrix §Reliability | `●` for 8 |
| Failure-injection / chaos suite | matrix §Reliability | `●` for 8 |
| Tool calling and typed tool schemas | matrix §LLM | `●` for 4 and 8 |
| Stopping conditions and permissions | matrix §LLM | `●` for 4 and 8 |
| Audit-event contract, independent implementation | matrix §HITL | `●` for 8 |
| Policy / permission enforcement | matrix §HITL | `●` for 8 |
| Groundedness / faithfulness scoring | matrix §Evaluation | `●` for 8 |
| Retrieval-vs-generation error attribution | matrix §Evaluation | `●` for 8 |
| Committed golden dataset + eval as a build-failing CI gate | matrix §Evaluation | `●` for 8 |
| Background workers, scheduled jobs | matrix §Backend, §Observability | `●` for 8 |

**The blueprint's own required tests:** a kill-and-resume test in CI *asserting resumption at the
same node with no tool re-execution*; an idempotent submission test; citation-groundedness scoring
against the committed golden set with retrieval-versus-generation attribution; deadline and
escalation unit tests.

**The blueprint's recorded risk, and what is done about it.** *"The warranty domain is less familiar
to the builder than the finance and insurance domains elsewhere in the slate. Fixture realism depends
on domain research that is not budgeted. Re-examine before build; the fallback is another
commercial-liability case domain with the same state shape."*

Re-examined. **The domain is kept, and the risk is discharged the way projects 6 and 7 discharged
theirs: the corpus is synthetic, generated from a committed seed, and described as synthetic in the
body of every generated file and on every screen.** No claim of realism is made that a reader could
check and find false. The engineering — a durable case machine over deterministic money and cited
evidence — does not depend on the fixtures being drawn from life, and the alternative domain would
have cost the same and proved nothing more. What is *not* claimed: that these policy texts resemble
any manufacturer's, that the rejection codes are an industry standard, or that the recovery rates
mean anything outside this corpus.

### 2. Where the brief and the authoritative sources diverged, and what was built

The task brief for this build framed the project as *"Is this warranty claim recoverable, from whom,
for how much, and on what evidence?"* — a deterministic eligibility-and-amount engine. The blueprint
frames it as a *durable case workflow that corrects and resubmits a rejected claim*. These are not
the same project, and the brief itself says the authoritative sources override it.

**Built: the blueprint's project, with the brief's deterministic core inside it.** The two unify
without strain, because a resubmission is only worth filing if the money behind it is right:

```
a manufacturer rejection arrives (code + case context)
  → deterministic requirement matrix   what evidence does this rejection code demand?
  → deterministic deadline arithmetic  is the manufacturer's claim window still open?
  → deterministic eligibility + money  warranty period, part eligibility, cap, deductible,
                                       prior recovery, currency
  → retrieval with citation            which policy clause or service bulletin governs this?
  → correction proposal                composed from cited spans; the model arm is a port
  → deterministic gate                 RECOVERABLE / PARTIALLY_RECOVERABLE /
                                       NOT_RECOVERABLE / REVIEW
  → HITL approval breakpoint           a person approves before anything leaves the system
  → idempotent submission              at most one effect per recovery identity
```

The brief's four recovery states are adopted verbatim, because the blueprint fixes no names and
four states are the right number: forcing an ambiguous claim to a binary outcome is the failure this
project exists to prevent.

**The deterministic core owns, and no model may touch:** claimed amount, eligible amount, excluded
amount, deductible, cap, recoverable amount, warranty period, part and serial eligibility,
supplier/manufacturer responsibility, claim-window deadlines, prior-recovery state, duplicate
detection, and the gate decision itself.

### 3. Divergences from the blueprint, recorded now rather than discovered later

| blueprint says | built | why |
|---|---|---|
| DigitalOcean App Platform + managed Postgres | **Render Free + Neon Free (pgvector) + Upstash Free (Redis)** | DigitalOcean App Platform has no free tier for a web service, and no payment authorisation exists. The portfolio's deployment-spread requirement is already met across seven other targets. Recorded as a divergence, not presented as the plan. |
| Next.js case console | **server-rendered HTML** unless a build step earns itself | Projects 5, 6 and 7 took the same decision; a build step that does not make the evidence on screen more legible is cost without benefit. Revisited at M7 against what the console actually needs. |
| "The model proposes a correction with a citation and a confidence" | **the shipped composer is extractive**; the abstractive arm is a port whose `propose` **raises** | No model credential exists in this environment. A stub, an echo or a canned string would enter the evaluation as a number attributed to a model that was never called — the exact defect project 7's ADR-001 forbids. The port, its typed tool schemas, its permission checks and its post-validation are built and tested against hand-written inputs, and `docs/` records what a key would enable. **No cost, latency or quality figure may be published for the abstractive arm.** |

### 4. The claims this project makes

1. **Durability.** Killing the worker mid-case resumes the case at the node the checkpoint recorded,
   re-executes no tool that had already completed, and loses no human decision.
2. **Nothing leaves without approval, and nothing leaves twice.** Every submission carries a recorded
   human approval for the same case and the same case version, and one recovery identity produces at
   most one submission effect under concurrency.
3. **Deterministic evidence decides the money.** The recoverable amount, the eligibility and the gate
   outcome are computed from source-backed facts with `Decimal` arithmetic, and every requirement the
   system reports as satisfied is traced to a verbatim span in a named document version.

### 5. The thirteen kill conditions

**Predeclared. Thresholds fixed here, before implementation.** Graded by
`tests/test_kill_criteria.py`, which reads `artifacts/*.json` and imports nothing from this package.
A criterion may be raised. None may be lowered, deleted, renamed, relaxed, skipped or `xfail`ed.

| | the project fails if | threshold | graded over |
|---|---|---|---|
| **A** | a resumed case continues at a different node from the one the checkpoint recorded | **0** divergences | every killed case, every node |
| **B** | a tool that completed before the kill runs again after the resume | **0** re-executions | every killed case |
| **C** | a human decision recorded before the kill is absent after the resume | **0** losses | every killed case |
| **D** | a submission appears in the audit log without a preceding approval for the same case **and** the same case version | **0** unapproved submissions | whole corpus |
| **E** | concurrent identical submissions produce more than one submission effect | **exactly 1** effect, **0** duplicates | ≥16 concurrent attempts per identity |
| **F** | a computed recoverable amount differs from the generator's ground truth | **0** mismatches, exact `Decimal` equality | whole corpus |
| **G** | a case is gated `RECOVERABLE` or `PARTIALLY_RECOVERABLE` whose ground truth is `NOT_RECOVERABLE` — a **false recovery** | **0** | hold-out |
| **H** | a requirement is reported satisfied with no citation | **0** | whole corpus |
| **I** | a cited span is not present verbatim at its recorded offsets in the document version it names | **0** | whole corpus |
| **J** | hold-out recall@5 for the governing clause is below **0.85**, or is not strictly above **every** predeclared baseline | **≥ 0.85** and **> max(baselines)** | hold-out |
| **K** | the dense retrieval stage's executed statement does not use the pgvector distance operator against a `vector` column, proven by the server's own `EXPLAIN` | operator present | live PostgreSQL |
| **L** | with Redis unavailable, a case is leased twice or a submission proceeds | **0** double-leases, **0** submissions | failure injection |
| **M** | a case is resubmitted after the manufacturer's claim window closed | **0** | whole corpus |

**The vacuity guard.** Every criterion above publishes the size of its own numerator and denominator
into its artifact, and `scripts/release_gate.py` **fails the build if any graded criterion's
denominator is zero**. Project 7's kill condition G passed because its numerator was empty by
construction, and the pass was worthless. A criterion that cannot fail is not a criterion, and this
project refuses to count one as a pass.

**Why K asks for the operator and not an index scan.** Project 7's kill condition K required a vector
*index scan* in the query plan, and failed because its own metadata filter left so few candidate rows
that PostgreSQL correctly preferred a sequential scan. The criterion asked for a plan that would have
been the wrong one. This project asks whether the extension is doing the arithmetic, which is the
claim actually being made.

**Why J's baselines differ in the retrieval stage itself.** Project 7's kill condition F required the
system to beat a baseline that removed only the gate and therefore ran the identical retriever — an
impossible target and a defect in the criterion. The baselines here are listed in §6 and each removes
or replaces a **retrieval** component, so beating them is a real question.

### 6. Predeclared baselines

Fixed before any score exists. Each is a genuinely different retrieval system, not this one with a
downstream component disabled.

| baseline | what it is |
|---|---|
| `exact_code_lookup` | resolve the governing clause by exact rejection-code → clause-id table lookup, no text search at all |
| `bm25_only` | lexical BM25 over the same clause corpus, no embeddings |
| `dense_no_metadata` | embeddings over the same corpus with **no** manufacturer/policy-version filter — retrieves from every manufacturer at once |
| `first_clause_of_policy` | always return the policy's first clause; the floor that says whether the task is trivial |

### 7. The hold-out

Split by **warranty program** — a manufacturer plus a policy version — never by claim. Claims under
one program share clause text, a rejection-code table and a deadline rule, so a claim-level split
measures memorisation of a policy the system was tuned on.

> a program is held out iff `blake2b(program_id, digest_size=8) % 100 < 30` (big-endian)

The rule consults no seed and no score. `artifacts/holdout.json` materialises the membership and its
digest, is committed in a commit that contains **no score**, and the harness raises if the corpus
would now produce a different split. **Nothing may be tuned against the hold-out after it has been
scored.** If a change is needed afterwards it is a new experiment with a new hold-out under a new
recorded decision.

### 8. What is deliberately out of scope

Recorded so it cannot look like an omission discovered later.

- **No live model call.** §3 explains why. No cost, latency or quality figure for the abstractive arm.
- **No real manufacturer portal.** Submission targets a local mock with a recorded contract; the
  idempotency claim is about this system's effects, not about a third party's behaviour.
- **No multi-agent architecture.** One graph, explicit nodes. The master spec forbids multi-agent as
  a README ornament.
- **No Kubernetes, no Kafka, no Temporal.** All three are recorded portfolio-wide omissions;
  `SKILL_MATRIX.md` names LangGraph's Postgres checkpointer as the deliberate answer to Temporal.
- **No OAuth.** Project 4 is the sole home for authorization; this project enforces an approver token
  and a policy, not an identity provider.

---

## ADR-002 — The hold-out was computed once before three defects were found, and what that means

**Status:** accepted · **Date:** 2026-09-28 · **ADR-001 stands unedited above.**

### What happened

On 2026-09-25 at 11:53 a build agent working on the evaluation lane ran the artifact builder in its
default mode, which scores **both** splits. The session it ran in was then interrupted, and the
resulting files were left on disk **uncommitted**. So the hold-out was computed once, by the
pre-fix system, and never entered git history — which is why `git log` shows no score artifact
before the scoring commit, and why this record exists: a clean history is not the same as a
hold-out nobody has seen, and a reader is owed the difference.

Those files are preserved verbatim in `artifacts/prior_run_2026-09-25/` rather than overwritten.
Overwriting them would have left the history clean and the account false.

### What the pre-fix run measured on the hold-out

| | |
|---|---|
| recoverable amount against ground truth | **0 mismatches** over 720 cases, exact `Decimal` equality |
| false recoveries (kill condition G) | **0** of 50 not-recoverable hold-out cases |
| false denials | 5 of 200 hold-out cases |
| review rate | 0.135 |
| resubmissions after the window closed (M) | 0 of 54 closed-window cases |
| retrieval recall@5, the system | **0.99** |
| retrieval recall@5, `bm25_only` | **1.00** |
| retrieval recall@5, `exact_code_lookup` | 0.85 |
| retrieval recall@5, `dense_no_metadata` | 0.34 |
| retrieval recall@5, `first_clause_of_policy` | 0.00 |
| candidates surviving the filter per query | 15, every query; 0% at or below k |

Its release gate read **FAILED on kill condition J**: the system cleared the 0.85 floor and was not
strictly above `bm25_only`. Conditions A to E were not graded, because the durability and
submission artifacts did not yet carry the keys the kill test reads.

### Three defects found afterwards, and how each was found

Each was found by reading the code or reasoning about the deployment, **not** by reading a score,
and each is provable without reference to one. `PORTFOLIO_MASTER_SPEC.md` and this project's brief
both allow exactly that class of fix after scoring, on condition that it is recorded — which is
what this section is.

1. **`case_version` was never incremented anywhere in the graph.** Found by searching for the
   increment and finding none. Every case therefore sat at version zero forever, and "an approval
   for the same case version" was satisfied trivially: a case re-priced after approval would still
   have matched the approval of its old amount. **It affects durability and submission. It moves
   no hold-out recovery or retrieval number.**
2. **Cross-case duplicate prevention rested on a store with no persistence.** Found by reading the
   plan of the deployed Key Value instance: Render's free Valkey has none, so any restart empties
   every idempotency marker, and a second case carrying the same physical recovery would then file
   it again. The durable backstop is `submission_record` with the recovery identity as its primary
   key, migration `0002`. **It affects submission. It moves no hold-out recovery or retrieval
   number.**
3. **Retrieval could cite a withdrawn bulletin.** Found by reading the retrieval statement, which
   filtered on programme and policy version only, and the loader's own note that nothing in the
   package decided anything on `is_current`. Confirmed against the generator's construction rather
   than against a score: **0 of 720 governing clauses point at a non-current document**, so
   excluding non-current documents before ranking cannot remove a single correct answer — it can
   only remove wrong ones.

### Why the third fix cannot be tuning to pass J, stated before the final score

This is the one that touches a retrieval number, so it is argued here rather than assumed.

Kill condition J requires hold-out recall@5 to be **strictly above every baseline**, and on this
hold-out `bm25_only` scored **1.00**. No system can be strictly above 1.00. The fix removes
superseded candidates from the dense system's pool; at best it raises the system from 0.99 to 1.00,
which is **equal** to the baseline and therefore still fails the criterion as ADR-001 wrote it.
The fix is incapable of flipping J, which is the property that makes it safe to apply after J was
seen to fail.

### What was deliberately not changed after the hold-out was seen

None of the following was touched, and none will be: the four baselines; kill condition J's
wording or threshold; the query composition; the embedding model; `k`; the corpus; the hold-out
membership; any gate rule; any money rule. A system that failed J because a lexical baseline
saturated the task is **not** repaired by making the baseline weaker or the criterion softer, and
doing either would be the precise failure this project's contract was written to make impossible.

### What J failing would mean

That on this synthetic corpus the retrieval task is **lexically trivial**: the rejection code's own
vocabulary appears in the clause that governs it, so term matching alone finds the governing clause
every time and embeddings add nothing measurable. It is the same class of lesson as project 7's —
a task easier than the criterion assumed — reached by a different route. Project 7's F failed
because its baseline ran the identical retriever; ADR-001 §6 was written to prevent exactly that,
and did. J's baselines are genuinely different systems. One of them simply wins.

That is a finding about the corpus and about when embeddings earn their place. It is not a finding
the project may argue its way out of.
