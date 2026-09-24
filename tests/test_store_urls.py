"""The four connection-URL shapes a managed PostgreSQL hands over, pinned, and the session scope.

Nothing here touches a server. `normalise_database_url` is string arithmetic and `session_scope` is
a transaction protocol, and both are tested where that can be done exhaustively and in milliseconds.
The tests that need PostgreSQL are in `test_retrieval.py`, which pays for a database once.

The four shapes are not hypothetical and each has a failure attached to it:

* `postgres://` — Render emits it; SQLAlchemy 2.0 has no `postgres` dialect and raises at
  `create_engine`, inside a worker's start-up, where the symptom is a restarting container.
* `postgresql://` — Neon prints it; SQLAlchemy resolves it to the default driver, `psycopg2`, which
  this project does not install, so the symptom is a missing-module error naming the wrong package.
* `postgresql+psycopg://` — already right, and must survive untouched.
* any of the above with `?sslmode=require` — the query string carries the security setting, and a
  normaliser that rebuilds the URL from parsed components drops it. The connection then succeeds,
  unencrypted, which is the only one of the four failures that does not look like a failure.

`session_scope` is exercised over an in-memory SQLite database. That is deliberate: the property
under test is "commit on success, roll back on every other exit", which is SQLAlchemy's session
protocol and not PostgreSQL's. Running it against a real server would make a fast, exhaustive test
into a slow, conditional one and would test the same three branches.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Column, Engine, Integer, MetaData, String, Table, create_engine, select

from warranty_claim_recovery.store.engine import (
    TARGET_DRIVER,
    UnsupportedDatabaseUrlError,
    build_engine,
    normalise_database_url,
    session_scope,
)

CREDENTIALS = "warranty:warranty_local_only@db.example.internal:5432/warranty_claim_recovery"


@pytest.mark.parametrize(
    "given",
    [
        f"postgres://{CREDENTIALS}",
        f"postgresql://{CREDENTIALS}",
        f"{TARGET_DRIVER}://{CREDENTIALS}",
    ],
)
def test_every_accepted_shape_normalises_to_the_installed_driver(given: str) -> None:
    assert normalise_database_url(given) == f"{TARGET_DRIVER}://{CREDENTIALS}"


@pytest.mark.parametrize("scheme", ["postgres", "postgresql", TARGET_DRIVER])
def test_the_query_string_survives_normalisation(scheme: str) -> None:
    """`sslmode=require` reaching the server is the difference between TLS and no TLS."""
    given = f"{scheme}://{CREDENTIALS}?sslmode=require&connect_timeout=10&application_name=wcr"
    normalised = normalise_database_url(given)
    assert normalised.endswith("?sslmode=require&connect_timeout=10&application_name=wcr")
    assert normalised.startswith(f"{TARGET_DRIVER}://")


def test_a_password_with_reserved_characters_is_copied_byte_for_byte() -> None:
    """The authority is never parsed, so it is never re-encoded.

    A normaliser built on `make_url(...).set(drivername=...)` round-trips the password through URL
    decoding and re-encoding, and a password containing a percent sign or a plus comes back changed.
    The authentication failure that follows is then attributed to the credential rather than to the
    parser that altered it.
    """
    password = "p%40ss+w/ord%2F!"
    given = f"postgres://warranty:{password}@host:5432/db"
    assert normalise_database_url(given) == f"{TARGET_DRIVER}://warranty:{password}@host:5432/db"


def test_an_explicit_driver_this_project_does_not_install_is_refused_not_rewritten() -> None:
    """`postgresql+psycopg2://` is a deliberate statement, and honouring it silently is wrong.

    Rewriting it would give the caller a different driver from the one they asked for, with nothing
    in the logs saying so. The error instead names both drivers, so the reader can decide.
    """
    with pytest.raises(UnsupportedDatabaseUrlError) as raised:
        normalise_database_url(f"postgresql+psycopg2://{CREDENTIALS}")
    message = str(raised.value)
    assert "psycopg2" in message
    assert TARGET_DRIVER in message


@pytest.mark.parametrize(
    "given",
    [
        "mysql://warranty:secret@host:3306/db",
        "sqlite:///./warranty.db",
        "postgresql+asyncpg://warranty:secret@host:5432/db",
    ],
)
def test_a_url_this_deployment_cannot_honour_raises_rather_than_being_coerced(given: str) -> None:
    with pytest.raises(UnsupportedDatabaseUrlError):
        normalise_database_url(given)


def test_a_string_that_is_not_a_url_at_all_says_so() -> None:
    with pytest.raises(UnsupportedDatabaseUrlError) as raised:
        normalise_database_url("warranty_claim_recovery")
    assert "://" in str(raised.value)


def test_build_engine_normalises_before_it_creates() -> None:
    """A `postgres://` URL must not reach `create_engine`, which raises `NoSuchModuleError`."""
    engine = build_engine(f"postgres://{CREDENTIALS}?sslmode=require")
    try:
        assert engine.url.drivername == TARGET_DRIVER
        assert engine.url.query["sslmode"] == "require"
    finally:
        engine.dispose()


def test_build_engine_refuses_an_unsupported_url_with_this_projects_error() -> None:
    with pytest.raises(UnsupportedDatabaseUrlError):
        build_engine("mysql://warranty:secret@host:3306/db")


# ------------------------------------------------------------------------------------------------
# session_scope. See the module docstring for why SQLite is the right database for these three.
# ------------------------------------------------------------------------------------------------

_METADATA = MetaData()
_NOTE = Table("note", _METADATA, Column("id", Integer, primary_key=True), Column("body", String))


def _sqlite_engine() -> Engine:
    engine = create_engine("sqlite://")
    _METADATA.create_all(engine)
    return engine


def _rows(engine: Engine) -> list[tuple[int, str]]:
    with session_scope(engine) as session:
        return [(row[0], row[1]) for row in session.execute(select(_NOTE.c.id, _NOTE.c.body))]


def test_a_scope_that_returns_normally_commits() -> None:
    engine = _sqlite_engine()
    with session_scope(engine) as session:
        session.execute(_NOTE.insert().values(id=1, body="approved"))
    assert _rows(engine) == [(1, "approved")]


def test_an_exception_rolls_the_whole_scope_back() -> None:
    engine = _sqlite_engine()
    with pytest.raises(RuntimeError), session_scope(engine) as session:
        session.execute(_NOTE.insert().values(id=1, body="half written"))
        raise RuntimeError("the manufacturer portal refused the submission")
    assert _rows(engine) == []


def test_a_kill_rolls_the_scope_back_although_it_is_not_an_exception() -> None:
    """The reason `session_scope` catches `BaseException`, as a test rather than as a comment.

    This project kills workers on purpose. An in-process kill arrives as `KeyboardInterrupt` or
    `SystemExit`, neither of which inherits from `Exception`, so a scope catching only `Exception`
    would return a connection to the pool inside an open transaction. Kill conditions A, B and C are
    all about what survives a kill, and a half-written case that another worker can observe is
    exactly what must not.
    """
    engine = _sqlite_engine()
    with pytest.raises(KeyboardInterrupt), session_scope(engine) as session:
        session.execute(_NOTE.insert().values(id=1, body="written just before the kill"))
        raise KeyboardInterrupt
    assert _rows(engine) == []
