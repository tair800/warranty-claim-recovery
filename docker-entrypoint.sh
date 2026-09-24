#!/bin/sh
# Migrate, load the precomputed index if the database is empty, then serve.
#
# **`set -e` is deliberately absent, and this is the most important line in the file.** A seeding
# failure must not crash-loop the container. The evidence screen is artifact-backed and works with
# no database at all, `/healthz` reports the real state of the database and the queue, and a console
# that serves what it can and explains the rest is worth more to a visitor — and to whoever is
# debugging — than one that restarts every thirty seconds. The failure is loud in the logs and
# visible at `/healthz`, which is where an operator looks. A container that exits instead takes its
# own logs with it.
#
# The migration runs here rather than as a separate release command because this platform has no
# release phase on the free plan. That is a trade: two containers starting at once would both run
# `alembic upgrade head`, and alembic's version table plus its transaction make the loser a no-op
# rather than a corruption. One web instance is what the free plan gives, so the race is theoretical
# today and survivable if it ever stops being.
#
# `WCR_SKIP_SEED=true` skips both database steps. CI passes it to prove the image starts with no
# database reachable at all — without it that step would really be measuring how long a connection
# timeout takes to expire.
#
# `exec` on the last line so uvicorn becomes PID 1 and receives SIGTERM directly. Without it the
# shell holds PID 1, swallows the signal, and every deploy waits out the platform's kill timeout
# before the old container dies.

if [ "${WCR_SKIP_SEED}" = "true" ]; then
  echo "[entrypoint] WCR_SKIP_SEED=true; skipping migration and indexing"
else
  echo "[entrypoint] bringing the schema up to head"
  (cd /app && python -m alembic upgrade head) \
    || echo "[entrypoint] migration failed; serving anyway, /healthz will say so"

  echo "[entrypoint] loading the precomputed clause index if the table is empty"
  # `--skip-if-populated` because this runs on every start. Re-embedding a populated index would
  # add minutes to each deploy and would rewrite rows the running evaluation was measured against.
  python /app/scripts/seed_index.py --skip-if-populated \
    || echo "[entrypoint] indexing failed; serving anyway, /healthz will say so"
fi

echo "[entrypoint] starting uvicorn on ${PORT:-8000}"
exec uvicorn warranty_claim_recovery.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
