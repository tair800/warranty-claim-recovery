# Two stages. The build stage has uv and a compiler; the runtime stage has neither. That is smaller,
# and it is also a smaller attack surface: nothing in the shipped image can resolve a dependency or
# compile a wheel, so an image that has been on a registry for a month contains exactly what the
# lock file said it contained when it was built.

FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

# A compiler, and the OpenMP runtime onnxruntime links against. The compiler is here rather than in
# the runtime stage because every wheel this lock file resolves today ships a binary for this
# platform — and the day one of them does not, the build should fail loudly on a missing toolchain
# here rather than at `pip install` time on a machine nobody expected to be compiling.
RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies before source, so that editing a Python file does not re-resolve the lock file.
# `--frozen` refuses to update it: an image whose dependencies drifted from uv.lock is not the thing
# the tests were run against, and the difference would only surface in production.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# `README.md` and `LICENSE` are here because `pyproject.toml` names both in its metadata, so
# hatchling reads them when it builds the project wheel. Without them the second sync fails on a
# missing file that nobody thinks of as source, and the error names the README rather than the
# packaging configuration that asked for it.
COPY README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

# The embedding model, baked in at build time.
#
# Roughly 130MB of ONNX weights. Fetching them on first request would make the first case a reviewer
# opens after a cold start wait on a model host, and would make this service depend for ever on that
# host being reachable. Baking them in trades image size for a dependency the deployment does not
# have. The model name is the one `artifacts/encoder_memory.json` was measured with; changing it
# here without re-measuring would invalidate the only number the deployment topology rests on.
ENV WCR_EMBEDDING_CACHE=/app/.fastembed_cache
RUN /app/.venv/bin/python -c "\
from fastembed import TextEmbedding; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/.fastembed_cache')"

# The corpus and its clause vectors, built here rather than at container start.
#
# The corpus is regenerated from its committed seed rather than copied in, because `data/generated`
# is gitignored on purpose: a corpus that lives in git history is a corpus nobody regenerates, and
# the seed then quietly stops being the source of truth. Rebuilding it here keeps the seed
# authoritative and keeps the generated JSON out of the repository.
#
# The clause embeddings are precomputed in the same layer, so the container loads vectors instead of
# computing them. Encoding the clause corpus takes minutes on a builder and would take considerably
# longer on a shared-CPU free instance — long past any platform health check — and a container that
# is still encoding when the health check gives up is a deploy that never goes live.
RUN /app/.venv/bin/python scripts/generate_corpus.py \
 && /app/.venv/bin/python scripts/precompute_embeddings.py


FROM python:3.12-slim-bookworm AS runtime

# The OpenMP runtime, and nothing else. onnxruntime's manylinux wheel links against libgomp.so.1,
# which python:3.12-slim does not carry; without it the first query fails with a loader error naming
# a shared object, which reads exactly like a missing model and sends whoever is on call looking in
# the wrong place. No compiler and no uv in this stage, deliberately.
RUN apt-get update \
 && apt-get install --no-install-recommends -y libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# A system user with no home directory and no shell. This process serves a public demo; it has no
# business being able to log in, and nothing it does legitimately needs to write outside /tmp and
# the embedding cache below.
RUN useradd --system --no-create-home --shell /usr/sbin/nologin --uid 10001 warranty

WORKDIR /app

COPY --from=build --chown=root:root /app/.venv /app/.venv
COPY --from=build --chown=root:root /app/src /app/src
COPY --from=build --chown=root:root /app/scripts /app/scripts

# Owned by the runtime user rather than by root, and this line is the whole reason the cache is
# called out separately. fastembed writes a small tree-cache file beside the weights the first time
# it opens them. Root-owned, that write fails, the library logs a permission error about a corrupted
# tree cache and then goes looking on the network for a model it already has — so every cold start
# fails in a way that reads as "the model is missing". That cost project 7 a deploy cycle. The model
# files remain effectively read-only: the process has no shell, and nothing writes here but
# fastembed's own bookkeeping.
COPY --from=build --chown=warranty:warranty /app/.fastembed_cache /app/.fastembed_cache

# The corpus and its precomputed clause vectors. The entrypoint loads these into PostgreSQL without
# opening the encoder at all, which is what lets a free-tier start finish in seconds.
COPY --from=build --chown=root:root /app/data/generated /app/data/generated

# The migrations. Without them the container could only bring up a schema through a test helper, and
# the thing deployed would not be the thing that was tested.
COPY --chown=root:root alembic.ini ./alembic.ini
COPY --chown=root:root alembic ./alembic

# The evidence. The evidence screen reads these files and nothing else, which makes it the one
# screen that works with no database at all — and it is the screen a reader should look at first.
# Leaving them out would deploy a console that reports "not measured" for every figure the README
# quotes, which is worse than deploying nothing.
COPY --chown=root:root artifacts ./artifacts

COPY --chown=root:root docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# `chmod` explicitly rather than trusting the mode git recorded. A shell script committed 100644
# from a Windows checkout lands non-executable wherever it is checked out next, and the container
# then exits 126 with a message that does not name the file. The git index mode is 100755 as well;
# this line means the image is correct even on the day that stops being true.
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV WCR_EMBEDDING_CACHE=/app/.fastembed_cache

# One thread for the ONNX session, and one for every BLAS library that would otherwise size a thread
# pool from the host's core count. `artifacts/encoder_memory.json` measures this encoder at 285MB
# resident against a 512MB instance, which is headroom but not much of it, and each additional
# thread takes its own arena. The batch path that encodes the whole clause corpus runs at build
# time, on a builder, where none of this applies.
ENV WCR_EMBEDDING_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Read-only unless a deployment says otherwise. A misconfigured public instance that refuses every
# mutation is an inconvenience; one that accepts them is an incident.
#
# Deliberately not set here: WCR_APPROVER_TOKEN and WCR_LLM_API_KEY. Both fail closed without a
# value, which is the point of them having no defaults, and an image that carried either would put
# the same credential on every deployment that ever ran it.
ENV WCR_READ_ONLY=true

USER warranty
EXPOSE 8000

# `/livez` answers whether the process is up and nothing else; `/healthz` answers whether the
# database and the queue are reachable, and truthfully says 503 when they are not. This probe asks
# the first question, because a container whose database is still starting is not a container that
# should be restarted. `PORT` is read from the environment because the platform assigns it —
# hard-coding 8000 would probe a port nothing is listening on.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/livez', timeout=4).status == 200 else 1)"

CMD ["/usr/local/bin/docker-entrypoint.sh"]
