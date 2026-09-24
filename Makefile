# Everything a reviewer needs, in the order they would run it.
#
#   make setup         dependencies
#   make db            PostgreSQL with pgvector on 127.0.0.1:15441, Redis on 127.0.0.1:16380
#   make migrate       alembic upgrade head
#   make corpus        the synthetic corpus, from its committed seed
#   make holdout       freeze the hold-out, before anything can be scored against it
#   make index         embed the clause corpus and load it into pgvector
#   make artifacts     run everything and write the evidence the kill test grades
#   make breaches      plant a defect into every guarantee and check each is caught
#   make test          the engineering suite
#   make release-gate  the thirteen predeclared kill conditions, reported
#   make console       the Warranty Recovery Lab on http://127.0.0.1:8081
#
# `make evidence` is the whole chain and is what CI runs.
#
# Four things in this file are deliberate rather than tidy, and each of them prevents a specific
# failure this portfolio has already met once.
#
# **`holdout` runs before `index` and before `artifacts`.** ADR-001 §7 fixes the hold-out before
# anything is scored against it. A Makefile whose chain froze the split after the evaluation ran
# would make the ordering claim depend on whoever typed the commands rather than on the build, and
# the first time someone regenerated a corpus to chase a recall number the split would follow it.
# The dependency list is the enforcement. Re-running the target is a verification rather than a
# re-draw, because the rule is a pure function of the program identifier that consults no seed and
# no score — `blake2b(program_id, digest_size=8) % 100 < 30` — so a second freeze either writes the
# identical membership or says that the corpus changed underneath it.
#
# **`release-gate` is not part of `test`, and `test` explicitly ignores the kill test.** They answer
# different questions. `test` asks whether the software works and must be green; the release gate
# asks whether the three claims in ADR-001 §4 hold, and its honest answer may be no. Merging them
# produces a build that is red for a disclosed reason, which is a build nobody reads, and a
# regression then hides behind the disclosure.
#
# **`fast` deselects the `infrastructure` marker rather than relying on those tests skipping.** A
# test that skips when it cannot reach PostgreSQL is doing the right thing, but a `fast` target
# built on that behaviour is a target that silently stops covering anything the day a connection
# string changes. Deselecting by marker says what is excluded; skipping only says what was absent.
#
# **`determinism` is a target and not a comment.** The corpus is generated from a committed seed and
# the hold-out digest is taken over it. If the generator is not byte-reproducible then the frozen
# split describes a corpus that no longer exists, and every score graded against it is graded
# against nothing.

.PHONY: help setup db db-down migrate corpus determinism holdout index artifacts breaches \
        test release-gate fast lint types console screenshots evidence clean

# Make would happily run the prerequisites of `evidence` in parallel under -j, and the chain is
# strictly ordered: the index needs the corpus, the artifacts need the index, and the hold-out must
# be frozen before either. Refusing parallelism here is cheaper than debugging a race that only
# appears on a machine with more cores than this one.
.NOTPARALLEL:

# The interpreter from the project's own virtual environment, never a bare `python`. A global
# interpreter that happens to import the package is an interpreter running a different resolution of
# the lock file, and the resulting "works on my machine" is unfalsifiable.
PY := .venv/Scripts/python.exe
ifneq ($(OS),Windows_NT)
PY := .venv/bin/python
endif

# The console's port. Chosen so it cannot collide with the other repositories in this workspace,
# for the same reason the compose file binds 15441 and 16380 rather than 5432 and 6379.
CONSOLE_PORT ?= 8081

help: ## list these targets
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## install dependencies into .venv from the committed lock
	uv sync --frozen

db: ## start PostgreSQL with pgvector and the Redis the queue and the leases run on
	docker compose up -d postgres redis

db-down: ## stop them
	docker compose down

migrate: ## bring the schema up to head
	$(PY) -m alembic upgrade head

corpus: ## generate the synthetic corpus from its committed seed
	$(PY) scripts/generate_corpus.py

determinism: ## build the corpus twice and diff every byte
	$(PY) scripts/generate_corpus.py --verify-determinism

holdout: ## freeze the hold-out, split by warranty program, before anything is scored
	$(PY) scripts/freeze_holdout.py

index: ## embed the clause corpus and load it into pgvector
	$(PY) scripts/seed_index.py

artifacts: ## run everything and write the evidence the kill test grades
	$(PY) scripts/build_artifacts.py

breaches: ## plant a defect into every guarantee and check each is caught
	$(PY) scripts/plant_breaches.py

lint: ## ruff, including the formatter as a check rather than as advice
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

types: ## mypy --strict over src
	$(PY) -m mypy --strict src

fast: lint types ## lint, types and every test that needs no infrastructure
	$(PY) -m pytest tests -q --ignore=tests/test_kill_criteria.py -m "not infrastructure"

test: ## the engineering suite -- everything except the predeclared kill test
	$(PY) -m pytest tests -q --ignore=tests/test_kill_criteria.py

release-gate: ## grade the thirteen predeclared kill conditions and report the result
	$(PY) scripts/release_gate.py

console: ## serve the Warranty Recovery Lab
	$(PY) -m uvicorn warranty_claim_recovery.api.app:app --port $(CONSOLE_PORT) --reload

screenshots: ## capture the console's screens into docs/screenshots/
	$(PY) scripts/screenshots.py

evidence: corpus determinism holdout migrate index artifacts breaches test release-gate ## the full chain, and what CI runs
	@echo
	@echo "The engineering suite is above it and the release gate is the last block. They are"
	@echo "separate on purpose: the suite says whether the software works, the gate says whether"
	@echo "ADR-001's three claims hold. Neither is allowed to stand in for the other."

clean: ## remove generated fixtures and tool caches
	rm -rf data/generated .pytest_cache .mypy_cache .ruff_cache
