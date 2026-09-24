"""The connection URL, and the four ways a managed PostgreSQL hands one over that do not work.

Nothing in this module is hypothetical. Every branch below exists because a specific provider emits
a specific string that SQLAlchemy 2.0 either refuses outright or resolves to a driver this project
does not install:

1. **Render emits `postgres://`.** SQLAlchemy removed the `postgres` dialect name in 1.4 and 2.0
   raises `NoSuchModuleError: Can't load plugin: sqlalchemy.dialects:postgres`. The failure happens
   at `create_engine`, which is usually inside a worker's start-up, so the first symptom is a
   container that restarts rather than an error anybody reads.
2. **Neon prints `postgresql://`.** SQLAlchemy accepts it and resolves it to the *default* driver
   for the dialect, which is `psycopg2`. This project installs `psycopg` (version 3) and not
   `psycopg2`, so the result is `ModuleNotFoundError: No module named 'psycopg2'` — an error that
   sends the reader looking for a missing dependency when the actual problem is a URL scheme.
3. **The query string carries the security setting.** Neon and Supabase both append
   `?sslmode=require`, and several well-meaning normalisers rebuild the URL from parsed components
   and drop it. The connection then still succeeds, unencrypted, against a database on the public
   internet. That is the worst of the four failures because nothing about it looks broken.
4. **An explicitly requested driver that is not installed.** `postgresql+psycopg2://` is a
   deliberate statement by whoever wrote it. Silently rewriting it to `psycopg` would honour a
   request nobody made; this module raises instead, and the message names what is installed.

So normalisation operates on the scheme and on nothing else. The authority, the path and the query
string are carried across as one opaque substring, because the only way to guarantee that
`?sslmode=require` survives is never to take it apart.

Rejected: `sqlalchemy.engine.make_url(url).set(drivername="postgresql+psycopg")`. It is the
idiomatic spelling and it round-trips the password through URL decoding and re-encoding. A password
containing a percent sign or a plus survives that trip changed, and the resulting authentication
failure is attributed to the credential rather than to the parser.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

__all__ = [
    "ACCEPTED_URL_SCHEMES",
    "TARGET_DRIVER",
    "UnsupportedDatabaseUrlError",
    "build_engine",
    "normalise_database_url",
    "session_scope",
]

#: The one driver this project installs. `pyproject.toml` pins `psycopg[binary]>=3.2`; psycopg2 is
#: not a dependency and adding it to satisfy a URL would be answering the wrong question.
TARGET_DRIVER: Final = "postgresql+psycopg"

#: The dialect names a managed provider may hand over. Both mean PostgreSQL; only one of them is a
#: name SQLAlchemy 2.0 still knows.
ACCEPTED_URL_SCHEMES: Final[frozenset[str]] = frozenset({"postgres", "postgresql"})

_SEPARATOR: Final = "://"


class UnsupportedDatabaseUrlError(ValueError):
    """A connection URL this project cannot honour, distinguished from one it cannot parse.

    A distinct type because the caller has somewhere specific to send the reader: not "bad URL" but
    "this names a database engine or a driver that this deployment does not carry". The deployment
    checklist can catch this type and print the supported schemes; a bare `ValueError` from three
    layers down cannot be caught that narrowly without also catching genuine parse failures.
    """


def normalise_database_url(url: str) -> str:
    """Coerce a provider's URL to the driver this project installs, keeping everything else.

    Only the substring before `://` is examined or rewritten. The remainder — user, password, host,
    port, database and **query string** — is copied verbatim, so `?sslmode=require` cannot be lost
    and a password is never decoded and re-encoded.

    Raises `UnsupportedDatabaseUrlError` for a non-PostgreSQL scheme and for an explicitly named
    driver other than `psycopg`, rather than rewriting either. See the module docstring for why a
    silent rewrite of an explicit driver choice is the wrong kindness.
    """
    scheme, separator, remainder = url.partition(_SEPARATOR)
    if not separator:
        raise UnsupportedDatabaseUrlError(
            f"{url!r} has no {_SEPARATOR!r} and is therefore not a connection URL at all; "
            f"expected something of the form postgresql://user:password@host:port/database"
        )

    dialect, _, driver = scheme.partition("+")
    if dialect not in ACCEPTED_URL_SCHEMES:
        raise UnsupportedDatabaseUrlError(
            f"{dialect!r} is not a PostgreSQL URL scheme. This project stores its clause vectors "
            f"in pgvector and its checkpoints in PostgreSQL, so no other engine is substitutable. "
            f"Accepted: {sorted(ACCEPTED_URL_SCHEMES)}."
        )
    if driver and driver != "psycopg":
        raise UnsupportedDatabaseUrlError(
            f"{scheme!r} asks for the {driver!r} driver and this project installs psycopg 3 only. "
            f"Rewriting the URL here would honour a request nobody made. Either install {driver!r} "
            f"deliberately, or drop the driver from the URL and let it normalise to "
            f"{TARGET_DRIVER!r}."
        )

    return f"{TARGET_DRIVER}{_SEPARATOR}{remainder}"


def build_engine(url: str) -> Engine:
    """An engine against a normalised URL, configured for a connection that goes away.

    `pool_pre_ping` is on because the deployment target is a free-tier managed PostgreSQL that
    closes idle connections from its side. Without the ping, the first statement after an idle
    period raises `OperationalError: server closed the connection unexpectedly`, and in this system
    that statement is usually the one a resumed case runs immediately after a worker picks it up —
    so the symptom presents as a durability bug rather than as a pool bug. The ping costs one
    round-trip per checkout, which is not measurable against an embedding call.

    `pool_recycle` is left at its default rather than guessed: the provider's idle timeout is not
    published, and a recycle interval longer than it is a number that looks like a fix and is not
    one. The pre-ping handles the case correctly whatever the timeout turns out to be.
    """
    return create_engine(normalise_database_url(url), pool_pre_ping=True, future=True)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """A session that commits on success and rolls back on **any** exit that is not success.

    The `except BaseException` is the point of this function and it is not overcautious. This
    project's headline claim is that a worker killed mid-case resumes correctly, and the test suite
    kills workers on purpose. A kill arrives in-process as `KeyboardInterrupt` or `SystemExit`,
    neither of which inherits from `Exception`; a scope that catches only `Exception` would let
    those propagate past the rollback. The connection then returns to the pool inside an open
    transaction, and SQLAlchemy's pool-level reset eventually discards it — but between those two
    moments a second session can observe work that the killed case had not finished. Rolling back
    explicitly removes the window.

    `expire_on_commit=False` so that rows read inside the scope remain readable after it. The
    default would emit a refresh query against a closed session and raise `DetachedInstanceError`,
    which reads like a lifecycle bug in the caller rather than the deliberate default it is.
    """
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
