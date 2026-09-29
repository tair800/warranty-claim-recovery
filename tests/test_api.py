"""The console's tests: what a reader is shown, and what they must never be shown.

`api/app.py` is the only module in this project that renders anything to a person who will never
read the code, and it is the only one whose defects are therefore visible to strangers. These tests
are organised around the failures that would matter on a deployed instance rather than around the
routes, because a test file laid out as one function per endpoint tends to check that each endpoint
returns 200 and to check nothing about the properties the endpoints exist to preserve.

**The leak.** A console is a template plus a configuration object, and the template that renders
`{{ settings.database_url }}` looks exactly like the template that does not. The same is true of
error text: a refused connection raises an exception whose message routinely carries the host, the
port and the user, so a page that prints `str(error)` publishes the DSN the first time PostgreSQL
is asleep. Both are checked here by planting sentinel strings — a database password, a user, a
database name, an approver token and a model key that appear nowhere else in this repository — into
the environment, requesting every screen, both health endpoints, the JSON route and all four
outcomes of the one mutating route, and searching the returned **bytes**. Searching the bytes rather
than the parsed text is deliberate: an escaped or JSON-encoded credential is still a credential, and
a test that parsed the HTML first would miss it inside an attribute.

**The verdict that is not the verdict.** The release-gate banner is on every screen, so a hard-coded
one would stay green on the day the gate went red, and a blank one reads as a pass to everyone who
has ever skimmed a table. Both are checked behaviourally: three different artifacts are written and
the three different banners are read back off the page, including a verdict string this codebase
does not contain, which no hard-coded template could produce. The rejected alternative was to grep
`base.html` for the literal `PASS`. It was rejected because `CLAUDE.md` §3.6 requires a guard to
fail from behaviour — a template can stop reading the artifact without any of its literals changing,
and the grep would still be green.

**Failing closed.** The one mutating route has three locks and each is checked on its own, including
the order they run in: a correct token in a read-only deployment must still be refused, and the
refusal must not depend on whether the claim exists, or the route would answer "which claims are
real" to a caller it has just turned away.

**Reporting rather than crashing.** `/healthz` is the one endpoint that must never raise, because a
500 from a readiness probe tells the platform that the health check is broken rather than that the
database is, and the operator then debugs the wrong process. It is tested against a refused
connection and against an injected fault, both with `raise_server_exceptions=False`, so that a crash
surfaces as an observable 500 and fails an assertion instead of erroring the test run with a
traceback nobody reads as a product defect.

These tests grade **no kill condition**. The thirteen are graded by `tests/test_kill_criteria.py`
from `artifacts/*.json`, and a console test that recomputed one would be a second grader — the exact
drift `api/app.py`'s own docstring rejects at length. What is asserted here is that the console
reports what the artifacts say, faithfully and without inventing a value when they say nothing.

The corpus used is the committed generated one rather than a fixture, because the screens' job is to
survive the real thing: seven hundred and twenty claims, eighteen of which cannot be priced at all.
A hand-made two-claim fixture would pass every assertion below and would not have found the
cross-currency case that renders a refusal instead of a total.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from markupsafe import escape

from warranty_claim_recovery.api import app as console
from warranty_claim_recovery.audit import APPROVAL_GRANTED
from warranty_claim_recovery.config import get_settings
from warranty_claim_recovery.money import Money

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATED_CORPUS = REPOSITORY_ROOT / "data" / "generated"
COMMITTED_ARTIFACTS = REPOSITORY_ROOT / "artifacts"

#: Strings that exist nowhere else in this repository. Each occupies a position in the DSN that a
#: driver error message is known to quote — psycopg names the user, the host, the port and the
#: database in a connection failure — so a message that reached a page would carry at least one of
#: them. A sentinel only in the password position would be a weaker test than it looks, because the
#: one field libpq is careful never to echo is the password.
SENTINEL_USER = "wcr_sentinel_user_ea41"
SENTINEL_PASSWORD = "wcr-sentinel-password-7b41c9"
SENTINEL_DATABASE = "wcr_sentinel_db_5c02"
SENTINEL_DSN = (
    f"postgresql+psycopg://{SENTINEL_USER}:{SENTINEL_PASSWORD}@127.0.0.1:59998/{SENTINEL_DATABASE}"
)
SENTINEL_TOKEN = "wcr-sentinel-approver-token-4d02fb"
SENTINEL_MODEL_KEY = "wcr-sentinel-model-key-11e9aa"
SENTINELS = (
    SENTINEL_USER,
    SENTINEL_PASSWORD,
    SENTINEL_DATABASE,
    SENTINEL_DSN,
    SENTINEL_TOKEN,
    SENTINEL_MODEL_KEY,
)

#: A DSN that refuses rather than hangs. 59998 is in the ephemeral range and nothing in this
#: project's compose file binds it, so the connection fails in milliseconds; an address that black
#: -holed instead would make every health test wait out `CONNECT_TIMEOUT_SECONDS`.
UNREACHABLE_DSN = "postgresql+psycopg://wcr_test:wcr_test@127.0.0.1:59998/wcr_absent"

SCREENS = ("/", "/evidence", "/recovery", "/evaluation", "/audit")

#: The two screens ADR-001's brief requires to work with no database at all: they read
#: `artifacts/*.json` and open no connection.
ARTIFACT_ONLY_SCREENS = ("/evaluation", "/audit")

#: Methods that cannot change anything. Everything else is a mutation and there must be one.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

#: The banner as `base.html` renders it: the tone class and the verdict text, captured separately so
#: a blank verdict is a failed assertion rather than an invisible one.
BANNER = re.compile(r'<span class="verdict ([a-z]+)">\s*([^<]*?)\s*</span>')

#: Any status pill with nothing inside it. A blank pill in a verdict column reads as a pass.
EMPTY_PILL = re.compile(r'<span class="pill(?: \w+)?">\s*</span>')

ENVIRONMENT_NAMES = (
    "WCR_DATABASE_URL",
    "WCR_REDIS_URL",
    "WCR_APPROVER_TOKEN",
    "WCR_LLM_API_KEY",
    "WCR_READ_ONLY",
    "WCR_CORPUS_DIR",
    "WCR_ARTIFACTS_DIR",
    "WCR_EMBEDDING_CACHE",
    "WCR_ENVIRONMENT",
)


class InjectedFaultError(RuntimeError):
    """A fault whose message carries every sentinel, planted where the console catches one.

    The point is the message. `_fault` is supposed to render the exception's **type name** and
    discard everything else, and the only way to prove that is to raise something whose message
    would be unmistakable on a page. Replacing a function the running system calls — rather than
    reading `app.py` and agreeing that it looks right — is what `CLAUDE.md` §3.6 requires of a
    guard, because a refactor that started printing `str(error)` would leave the source looking
    similar and would change the behaviour completely.
    """


def raise_injected_fault(*_: object, **__: object) -> Any:
    raise InjectedFaultError(
        f"could not connect to {SENTINEL_DSN} as {SENTINEL_USER} "
        f"with password {SENTINEL_PASSWORD} for database {SENTINEL_DATABASE}"
    )


@pytest.fixture(autouse=True)
def isolate_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Every test starts from a known environment and an empty artifacts directory.

    `get_settings` is memoised for the life of the process, so a test that sets an environment
    variable and does not clear the cache configures nothing and passes for the wrong reason. The
    cache is cleared on the way in and on the way out: on the way out because the next test in the
    file would otherwise inherit this one's settings through the cache even after `monkeypatch` has
    restored the environment itself.

    The artifacts directory is redirected to an empty temporary one rather than left pointing at the
    committed `artifacts/`. Other lanes in this repository write those files, and a banner assertion
    that depended on whether `release_gate.json` happened to exist would pass or fail according to
    what somebody else had run — which is a test that measures the working tree rather than the
    console. One test below deliberately points back at the real directory, and it asserts only that
    the screens render, which is true whatever the files contain.
    """
    for name in ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setenv("WCR_DATABASE_URL", UNREACHABLE_DSN)
    monkeypatch.setenv("WCR_CORPUS_DIR", str(GENERATED_CORPUS))
    monkeypatch.setenv("WCR_ARTIFACTS_DIR", str(artifacts))
    get_settings.cache_clear()
    yield artifacts
    get_settings.cache_clear()


def reconfigure(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    """Set environment variables and drop the memoised settings, in that order and always both."""
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()


def client(*, crashes_become_500: bool = False) -> TestClient:
    """A test client over the console.

    `crashes_become_500` exists for the health and approval tests. Starlette's default is to
    re-raise an unhandled exception into the test, which turns "this endpoint crashed" into a
    traceback rather than into a failed assertion about a status code. For the one endpoint whose
    contract is *it answers rather than raises*, the useful failure message is `assert 500 == 503`.
    """
    return TestClient(console.app, raise_server_exceptions=not crashes_become_500)


def write_artifact(directory: Path, name: str, payload: dict[str, Any]) -> None:
    directory.joinpath(name).write_text(json.dumps(payload), encoding="utf-8")


def as_rendered(value: str) -> str:
    """A string as Jinja will have written it into the page.

    Autoescaping turns an apostrophe into `&#39;`, so a test that compared a sentence taken from the
    code against the page's bytes would fail on the punctuation rather than on the substance — and
    the obvious repair, trimming the apostrophe out of the sentence, would weaken the assertion to
    hide a problem that was never there. Escaped with the template engine's own function rather than
    with `html.escape`, which spells the same character `&#x27;`; two spellings of "escaped" is one
    more thing that can drift.
    """
    return str(escape(value))


def banner_of(body: str) -> tuple[str, str]:
    """The banner's tone and verdict, or a failure that says the banner was not rendered at all."""
    found = BANNER.search(body)
    assert found is not None, "no release-gate banner was rendered on this screen"
    return found.group(1), found.group(2)


# ------------------------------------------------------------------------------------------------
# The corpus the screens are exercised against. Module-scoped: `load_corpus` memoises on the
# directory, so the parse and the clause-offset verification happen once for the whole file.
# ------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> console.Corpus:
    status = console.load_corpus(str(GENERATED_CORPUS))
    if status.corpus is None:
        pytest.fail(
            f"the generated corpus could not be read from {GENERATED_CORPUS} ({status.problem}). "
            f"Run `make corpus`; these tests deliberately use the committed corpus rather than a "
            f"fixture, because a two-claim fixture would not contain the cases that break screens."
        )
    return status.corpus


@pytest.fixture(scope="module")
def priced_claim(corpus: console.Corpus) -> str:
    """A claim the console can price all the way to a gate decision."""
    for claim_id in corpus.claim_order:
        case = console.build_case(corpus, claim_id)
        if case is not None and case.problem is None:
            return claim_id
    pytest.fail("no claim in the corpus could be priced, so every case screen test is vacuous")


@pytest.fixture(scope="module")
def unpriceable_claim(corpus: console.Corpus) -> str:
    """A claim the deterministic core refuses to price — the corpus's cross-currency cases.

    Found rather than hard-coded. A hard-coded identifier would quietly stop testing the refusal the
    day the generator renumbered its claims, and the test would still be green.
    """
    for claim_id in corpus.claim_order:
        case = console.build_case(corpus, claim_id)
        if case is not None and case.problem is not None:
            return claim_id
    pytest.fail(
        "no claim in the corpus is unpriceable, so the refusal path on the recovery screen is "
        "graded over an empty population and could not have failed"
    )


# ------------------------------------------------------------------------------------------------
# The five screens, and the two health endpoints.
# ------------------------------------------------------------------------------------------------


def test_every_screen_renders_without_a_database(priced_claim: str) -> None:
    """The evidence is on disk, so a database that is asleep costs the reader nothing."""
    http = client()
    for path in SCREENS:
        response = http.get(path, params={"claim": priced_claim})
        assert response.status_code == 200, f"{path} returned {response.status_code}"
        assert response.headers["content-type"].startswith("text/html")
        assert response.content, f"{path} rendered an empty body"


def test_the_artifact_screens_need_neither_a_database_nor_a_corpus(tmp_path: Path) -> None:
    """`/evaluation` and `/audit` read `artifacts/*.json` and nothing else.

    Pointed at an empty corpus directory as well as an unreachable database, because a screen that
    silently depended on the corpus would still have rendered in the test above.
    """
    empty = tmp_path / "no-corpus"
    empty.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        reconfigure(patch, WCR_CORPUS_DIR=str(empty))
        http = client()
        for path in ARTIFACT_ONLY_SCREENS:
            assert http.get(path).status_code == 200


def test_the_screens_render_against_the_committed_artifacts(priced_claim: str) -> None:
    """A smoke test against the real `artifacts/` directory, whatever it currently holds.

    It asserts nothing about the numbers on purpose. The committed artifacts are written by other
    lanes and their contents change; what must not change is that the console renders them rather
    than failing on a shape it did not expect.
    """
    with pytest.MonkeyPatch.context() as patch:
        reconfigure(patch, WCR_ARTIFACTS_DIR=str(COMMITTED_ARTIFACTS))
        http = client()
        for path in SCREENS:
            assert http.get(path, params={"claim": priced_claim}).status_code == 200


def test_every_screen_states_that_the_corpus_is_synthetic() -> None:
    """`CLAUDE.md` §3.8: synthetic, and said so on every screen, not only in the README."""
    http = client()
    for path in SCREENS:
        body = http.get(path).text
        assert "SYNTHETIC CORPUS." in body, f"{path} has no synthetic-corpus footer"
        assert console.SYNTHETIC_NOTICE in body, f"{path} renders a footer with different wording"


def test_livez_is_always_200_and_takes_no_dependency(tmp_path: Path) -> None:
    """Liveness answers with no database, no corpus and no artifacts.

    The platform's health check points here. A check wired to readiness instead would refuse to
    route traffic to a console that is serving five working screens, because its database had not
    finished waking up.
    """
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        reconfigure(
            patch,
            WCR_DATABASE_URL=UNREACHABLE_DSN,
            WCR_CORPUS_DIR=str(empty),
            WCR_ARTIFACTS_DIR=str(empty),
        )
        response = client(crashes_become_500=True).get("/livez")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_healthz_answers_503_rather_than_crashing_when_the_database_is_absent() -> None:
    """Readiness reports the database as not ready. It never reports itself as broken."""
    response = client(crashes_become_500=True).get("/healthz")
    assert response.status_code == 503, "a readiness probe that 500s blames the wrong process"
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["database"] == "unreachable"
    assert payload["screens_available_without_a_database"] == list(ARTIFACT_ONLY_SCREENS)


def test_healthz_reports_an_injected_fault_as_a_type_name_and_never_as_a_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe survives anything the engine can throw, and quotes none of it.

    The fault is planted by replacing `_engine`, which the running probe calls, rather than by
    reading the source and agreeing it looks careful.
    """
    monkeypatch.setattr(console, "_engine", raise_injected_fault)
    response = client(crashes_become_500=True).get("/healthz")
    assert response.status_code == 503
    assert response.json()["detail"] == "InjectedFaultError"
    for sentinel in SENTINELS:
        assert sentinel.encode() not in response.content


def test_healthz_returns_200_when_the_probe_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half of the readiness contract, so the 503 above is not simply the only answer."""
    monkeypatch.setattr(
        console, "database_health", lambda: console.DatabaseHealth(True, "SELECT 1 answered")
    )
    response = client(crashes_become_500=True).get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


# ------------------------------------------------------------------------------------------------
# Credentials. The sentinel sweep.
# ------------------------------------------------------------------------------------------------


def sweep(http: TestClient, claim_id: str) -> dict[str, bytes]:
    """Every response a reader or a scraper could obtain, as raw bytes.

    Includes both refusals and the accepted approval, because the response a caller gets **after**
    presenting the right credential is the one most likely to echo it back, and it is the one a
    reviewer never looks at.
    """
    responses: dict[str, bytes] = {}
    for path in (
        "/",
        "/evidence",
        "/recovery",
        "/evaluation",
        "/audit",
        "/livez",
        "/healthz",
        "/openapi.json",
        f"/?claim={claim_id}",
        f"/evidence?claim={claim_id}&retrieve=true",
        f"/recovery?claim={claim_id}",
        f"/api/case/{claim_id}",
        "/api/case/no-such-claim-at-all",
    ):
        responses[f"GET {path}"] = http.get(path).content

    approval = f"/api/case/{claim_id}/approval"
    body = {"case_version": 1, "note": "sentinel sweep"}
    responses[f"POST {approval} (no header)"] = http.post(approval, json=body).content
    responses[f"POST {approval} (wrong)"] = http.post(
        approval, json=body, headers={console.APPROVER_HEADER_NAME: "not-the-token"}
    ).content
    responses[f"POST {approval} (right)"] = http.post(
        approval, json=body, headers={console.APPROVER_HEADER_NAME: SENTINEL_TOKEN}
    ).content
    return responses


def test_no_configured_credential_reaches_any_response(
    monkeypatch: pytest.MonkeyPatch, priced_claim: str
) -> None:
    """The whole point of `PageHeader`, checked over the bytes rather than over the code.

    The environment carries a sentinel DSN whose user, password and database name appear nowhere
    else in this repository, a sentinel approver token and a sentinel model key. Mutations are
    enabled, so the accepted approval is exercised too. If any template ever gains
    `{{ settings.database_url }}`, or any handler starts rendering `str(error)`, one of these
    strings appears in one of these bodies and this assertion names the response it appeared in.
    """
    reconfigure(
        monkeypatch,
        WCR_DATABASE_URL=SENTINEL_DSN,
        WCR_APPROVER_TOKEN=SENTINEL_TOKEN,
        WCR_LLM_API_KEY=SENTINEL_MODEL_KEY,
        WCR_READ_ONLY="false",
    )
    responses = sweep(client(), priced_claim)
    assert responses, "the sweep requested nothing, so it proved nothing"
    for label, content in responses.items():
        for sentinel in SENTINELS:
            assert sentinel.encode() not in content, (
                f"{sentinel!r} leaked into the response to {label}"
            )


def test_no_credential_reaches_a_response_when_every_call_faults(
    monkeypatch: pytest.MonkeyPatch, priced_claim: str
) -> None:
    """The same sweep with the engine replaced by something that raises the credentials.

    A refused TCP connection produces a message that happens to be terse. This test removes the
    luck: every path that touches the database now raises an exception whose message is the DSN in
    full, and the pages must still carry a type name and nothing else.
    """
    reconfigure(
        monkeypatch,
        WCR_DATABASE_URL=SENTINEL_DSN,
        WCR_APPROVER_TOKEN=SENTINEL_TOKEN,
        WCR_LLM_API_KEY=SENTINEL_MODEL_KEY,
        WCR_READ_ONLY="false",
    )
    monkeypatch.setattr(console, "_engine", raise_injected_fault)
    responses = sweep(client(crashes_become_500=True), priced_claim)
    for label, content in responses.items():
        for sentinel in SENTINELS:
            assert sentinel.encode() not in content, (
                f"{sentinel!r} leaked into the response to {label}"
            )
    evidence = responses[f"GET /evidence?claim={priced_claim}&retrieve=true"]
    assert b"InjectedFaultError" in evidence, (
        "the retrieval failure was swallowed entirely; a reader needs to be told the stage did not "
        "run, and the type name is what they are told"
    )


def test_the_header_carries_only_the_two_fields_it_renders() -> None:
    """`page_header` is a value object, not a `Settings`, and this is what that buys.

    Asserted over the fields rather than over a rendered page, because the leak this prevents is a
    template gaining access to an object, not a template printing a particular string today.
    """
    header = console.page_header(get_settings())
    assert header._fields == ("environment", "read_only")


# ------------------------------------------------------------------------------------------------
# The banner. Read from the artifact on every screen, never inferred, never blank.
# ------------------------------------------------------------------------------------------------


def test_an_absent_release_gate_renders_not_run_on_every_screen() -> None:
    """A criterion nothing has graded is the one case a reader must not be left to assume about."""
    http = client()
    for path in SCREENS:
        tone, verdict = banner_of(http.get(path).text)
        assert verdict == console.NOT_RUN, f"{path} rendered {verdict!r} with no artifact present"
        assert tone == "unknown"
        assert verdict.strip(), "a blank verdict cell reads as a pass"


@pytest.mark.parametrize(
    ("payload", "expected", "tone"),
    [
        ({"verdict": "FAIL", "summary": "criterion G failed"}, "FAIL", "fail"),
        ({"verdict": "PASS", "summary": "thirteen of thirteen"}, "PASS", "pass"),
        ({"passed": False}, "FAIL", "fail"),
        ({"passed": True}, "PASS", "pass"),
        # A verdict no template in this repository contains. A hard-coded banner cannot produce it,
        # so reading it back off the page is proof the artifact is what is rendered.
        ({"verdict": "pass with reservations"}, "PASS WITH RESERVATIONS", "unknown"),
        # Present, parseable, and carrying no verdict. Distinct from absent because the remedies
        # differ: one is "run the gate", the other is "the gate wrote a shape this console does not
        # know how to read", and a reader who cannot tell them apart reruns the build for nothing.
        ({"note": "no verdict key here"}, console.UNREADABLE, "unknown"),
    ],
)
def test_the_banner_is_read_from_the_artifact_on_every_screen(
    isolate_configuration: Path, payload: dict[str, Any], expected: str, tone: str
) -> None:
    write_artifact(isolate_configuration, "release_gate.json", payload)
    http = client()
    for path in SCREENS:
        body = http.get(path).text
        assert banner_of(body) == (tone, expected), f"{path} disagreed with the artifact"
        if expected != "PASS":
            assert 'class="verdict pass"' not in body, (
                f"{path} rendered a passing banner while the artifact said {expected}"
            )


def test_a_malformed_release_gate_artifact_renders_not_run(isolate_configuration: Path) -> None:
    """Unparseable is the same answer as absent, and it is never a guess at the readable half."""
    isolate_configuration.joinpath("release_gate.json").write_text("{not json", encoding="utf-8")
    assert banner_of(client().get("/evaluation").text) == ("unknown", console.NOT_RUN)


# ------------------------------------------------------------------------------------------------
# The evaluation screen.
# ------------------------------------------------------------------------------------------------


def test_all_thirteen_criteria_are_listed_with_a_verdict_and_none_is_blank(
    isolate_configuration: Path,
) -> None:
    """The table is the thirteen, always, whatever the gate did or did not grade.

    A criterion that quietly stopped being listed is a criterion nobody notices stopped being
    graded, so the letters are asserted rather than the row count.
    """
    write_artifact(
        isolate_configuration,
        "release_gate.json",
        {"verdict": "FAIL", "criteria": {"A": "PASS", "G": {"verdict": "FAIL"}, "L": True}},
    )
    body = client().get("/evaluation").text

    letters = [criterion.letter for criterion in console.CRITERIA]
    assert letters == list("ABCDEFGHIJKLM")
    for criterion in console.CRITERIA:
        assert f"<strong>{criterion.letter}</strong>" in body
        assert as_rendered(criterion.fails_if) in body

    assert EMPTY_PILL.search(body) is None, "a verdict cell rendered empty, which reads as a pass"
    # Three letters were graded; the other ten must say NOT RUN rather than nothing at all.
    assert body.count(f">{console.NOT_RUN}<") >= len(console.CRITERIA) - 3
    assert '<span class="pill ok">PASS</span>' in body
    assert '<span class="pill bad">FAIL</span>' in body


def test_false_recoveries_are_their_own_number_and_are_not_folded_into_a_score(
    isolate_configuration: Path,
) -> None:
    """Four distinct figures, four distinct values on the page.

    Each measurement is given a value no other one shares, so a screen that averaged any two of them
    into a single score could not render all four. There is no number of correct recoveries that
    pays for one claim filed against a manufacturer on a basis that does not exist, and a combined
    score is precisely the arithmetic that says otherwise.
    """
    write_artifact(
        isolate_configuration,
        "recovery.json",
        {
            "cases_scored": 702,
            "amount_mismatches": 41,
            "holdout_false_recoveries": 17,
            "holdout_false_denials": 23,
            "holdout_review_rate": 0.29,
            "holdout_not_recoverable_cases": 88,
        },
    )
    body = client().get("/evaluation").text
    assert "False recoveries" in body
    assert "False denials" in body
    assert "REVIEW rate" in body
    for value in (">\n      17\n", "23", "41", "0.29", "88", "702"):
        assert value.strip() in body
    # The four counts must appear as four different strings, not as one derived figure.
    assert len({"17", "23", "41", "0.29"} & set(re.findall(r"[\d.]+", body))) == 4


def test_an_unmeasured_figure_is_not_run_and_never_a_zero() -> None:
    """`0` is a measurement; an absent key is not, and printing one as the other is a lie.

    A criterion that was never evaluated reported as `0` reads as a criterion that found nothing
    wrong, which is the most expensive misreading available on this screen.
    """
    body = client().get("/evaluation").text
    assert console.NOT_RUN in body
    assert console.metric(None, "holdout_false_recoveries") == console.NOT_MEASURED
    assert console.metric({"holdout_false_recoveries": 0}, "holdout_false_recoveries") == "0"


def test_the_retrieval_table_reports_the_system_and_every_baseline(
    isolate_configuration: Path,
) -> None:
    """Baselines are shown beside the system, because "beats every baseline" needs the baselines."""
    write_artifact(
        isolate_configuration,
        "retrieval.json",
        {
            "holdout_queries": 195,
            "k": 5,
            "system": {"recall_at_k": 0.93},
            "baselines": {
                "exact_code_lookup": {"recall_at_k": 0.61},
                "bm25_only": {"recall_at_k": 0.72},
                "dense_no_metadata": {"recall_at_k": 0.55},
                "first_clause_of_policy": {"recall_at_k": 0.14},
            },
        },
    )
    body = client().get("/evaluation").text
    for name in ("exact_code_lookup", "bm25_only", "dense_no_metadata", "first_clause_of_policy"):
        assert name in body
    for score in ("0.93", "0.61", "0.72", "0.55", "0.14"):
        assert score in body
    assert "195" in body


# ------------------------------------------------------------------------------------------------
# The case, evidence and recovery screens.
# ------------------------------------------------------------------------------------------------


def test_the_case_screen_shows_the_claim_the_rejection_and_the_window(
    corpus: console.Corpus, priced_claim: str
) -> None:
    body = client().get("/", params={"claim": priced_claim}).text
    record = corpus.claims[priced_claim]
    claim = record.claim
    assert claim.claim_id in body
    assert claim.part_number in body
    assert claim.failure_date.isoformat() in body
    assert claim.rejection_code.value in body
    assert claim.recovery_identity in body
    window = console.build_case(corpus, priced_claim)
    assert window is not None and window.window is not None
    assert window.window.closes_on.isoformat() in body
    assert str(abs(window.window.days_remaining)) in body
    assert window.decision is not None
    assert window.decision.outcome.value in body
    assert as_rendered(window.decision.reason) in body


def test_the_evidence_screen_shows_each_requirement_with_its_status_and_clause(
    corpus: console.Corpus, priced_claim: str
) -> None:
    """A requirement, its status, and the clause that satisfies it — with character offsets.

    The offsets are the claim: a citation is a verbatim span at a stated position in a stated
    document version, and a screen that printed only the quote would be showing something nobody
    could check against the source.
    """
    case = console.build_case(corpus, priced_claim)
    assert case is not None
    assert case.requirements, "this claim raised no requirement, so the table would be empty"
    body = client().get("/evidence", params={"claim": priced_claim}).text
    for requirement in case.requirements:
        assert requirement.requirement_id in body
        assert requirement.status.value in body
        citation = requirement.citation
        if citation is not None:
            assert citation.clause_id in body
            assert f"chars [{citation.start_offset}, {citation.end_offset})" in body
            assert citation.document_id in body
            assert citation.policy_version in body


def test_the_evidence_screen_does_not_run_retrieval_until_it_is_asked(priced_claim: str) -> None:
    """The empty state, and the reason it is an empty state rather than a slow first paint."""
    body = client().get("/evidence", params={"claim": priced_claim}).text
    assert "Not run on this page view." in body
    assert "Run retrieval" in body


def test_a_failed_retrieval_is_a_stated_error_and_not_a_stack_trace(
    monkeypatch: pytest.MonkeyPatch, priced_claim: str
) -> None:
    """The error state. The deterministic half of the screen must survive it intact."""
    monkeypatch.setattr(console, "_engine", raise_injected_fault)
    response = client(crashes_become_500=True).get(
        "/evidence", params={"claim": priced_claim, "retrieve": "true"}
    )
    assert response.status_code == 200
    body = response.text
    assert "The dense stage could not run." in body
    assert "InjectedFaultError" in body
    assert "What" in body and "demands" in body, "the offline half of the screen disappeared"


def test_the_recovery_column_is_a_column_that_adds_up(corpus: console.Corpus) -> None:
    """Every `=` line equals the lines above it, worked out the way a person would add an invoice.

    This is the whole reason the recovery screen is a column rather than a total: an adjudicator who
    cannot reconstruct the subtraction cannot check the figure, and the package they cannot check is
    the one they refuse. Sampled deterministically by stride rather than at random, because a
    sampled test that chose different cases on every run would fail on somebody else's machine and
    pass on the author's.
    """
    sampled = 0
    for claim_id in corpus.claim_order[::41]:
        case = console.build_case(corpus, claim_id)
        assert case is not None
        if case.problem is not None or case.computation is None:
            continue
        sampled += 1
        running = Money.zero(case.computation.currency)
        for row in case.rows:
            if row.operator == "":
                running = row.amount
            elif row.operator == "-":
                running = running - row.amount
            elif row.operator == "+":
                running = running + row.amount
            elif row.operator == "=":
                assert running == row.amount, f"{claim_id}: {row.label} does not follow"
                running = row.amount
        assert case.rows[-1].amount == case.computation.recoverable_amount
    assert sampled >= 1, "no case was sampled, so this test could not have failed"


def test_the_recovery_screen_renders_every_line_of_the_arithmetic(
    corpus: console.Corpus, priced_claim: str
) -> None:
    case = console.build_case(corpus, priced_claim)
    assert case is not None and case.computation is not None
    body = client().get("/recovery", params={"claim": priced_claim}).text
    for row in case.rows:
        assert row.label in body, f"the {row.label!r} line is missing from the screen"
    for label in (
        "claimed total",
        "labour excess",
        "uncovered parts",
        "eligible amount",
        "deductible",
        "removed by the claim cap",
        "already recovered",
        "recoverable amount",
    ):
        assert label in body


def test_a_case_that_cannot_be_priced_is_refused_rather_than_zeroed(
    unpriceable_claim: str,
) -> None:
    """A cross-currency claim has no recoverable amount this system is entitled to state.

    The honest screen is the refusal. A converted figure would look exact and would already be
    wrong, and a zero would be indistinguishable from a claim that genuinely recovers nothing.
    """
    body = client().get("/recovery", params={"claim": unpriceable_claim}).text
    assert "This case cannot be priced." in body
    assert "no exchange rate" in body

    payload = client().get(f"/api/case/{unpriceable_claim}").json()
    assert payload["problem"] is not None
    assert payload["computation"] is None
    assert payload["decision"] is None


def test_a_missing_corpus_is_an_explained_empty_state_and_not_a_traceback(tmp_path: Path) -> None:
    """A reviewer reading a traceback cannot tell a missing `make corpus` from a broken parser."""
    empty = tmp_path / "gone"
    empty.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        reconfigure(patch, WCR_CORPUS_DIR=str(empty))
        http = client(crashes_become_500=True)
        for path in ("/", "/evidence", "/recovery"):
            response = http.get(path)
            assert response.status_code == 200, f"{path} crashed instead of explaining itself"
            assert "The generated corpus could not be read." in response.text
            assert "make corpus" in response.text
        assert http.get("/api/case/anything").status_code == 503


def test_an_unknown_claim_falls_back_to_a_case_rather_than_a_status_code() -> None:
    """On a screen. `/api/case/{claim_id}` is an API contract and 404s; a screen is not one."""
    response = client().get("/", params={"claim": "no-such-claim"})
    assert response.status_code == 200
    assert "No case selected." not in response.text


# ------------------------------------------------------------------------------------------------
# The JSON route.
# ------------------------------------------------------------------------------------------------


def floats_in(node: Any, trail: str = "$") -> list[str]:
    """Every path in a parsed payload that holds a float. Money must never be one."""
    if isinstance(node, bool):
        return []
    if isinstance(node, float):
        return [trail]
    if isinstance(node, dict):
        return [
            found for key, value in node.items() for found in floats_in(value, f"{trail}.{key}")
        ]
    if isinstance(node, list):
        return [
            found for i, value in enumerate(node) for found in floats_in(value, f"{trail}[{i}]")
        ]
    return []


def test_the_case_json_carries_no_float_anywhere(priced_claim: str) -> None:
    """`Money.as_json` and never a float, at the one boundary where another system reads these.

    A float here would reintroduce binary floating point at exactly the point kill condition F
    grades exact `Decimal` equality, and it would do it silently: `json.dumps(0.1 + 0.2)` is a
    perfectly well-formed number.
    """
    payload = client().get(f"/api/case/{priced_claim}").json()
    assert payload["computation"] is not None
    assert floats_in(payload) == []
    assert isinstance(payload["computation"]["recoverable_amount"]["amount"], str)
    assert payload["is_synthetic"] is True
    assert payload["notice"] == console.SYNTHETIC_NOTICE


def test_the_case_json_404s_on_an_unknown_claim() -> None:
    response = client().get("/api/case/definitely-not-a-claim")
    assert response.status_code == 404
    assert response.json()["error"] == "unknown_claim"


# ------------------------------------------------------------------------------------------------
# Read-only by construction, and the one route that is not.
# ------------------------------------------------------------------------------------------------


def mutating_routes() -> list[str]:
    found: list[str] = []
    for route in console.app.routes:
        methods = set(getattr(route, "methods", None) or ())
        unsafe = sorted(methods - SAFE_METHODS)
        if unsafe:
            found.append(f"{','.join(unsafe)} {getattr(route, 'path', '?')}")
    return sorted(found)


def test_exactly_one_route_in_the_console_can_mutate_anything() -> None:
    """Read-only by construction means countable, not asserted in a README.

    A second mutating route added later fails here by name, which is the point: the claim on the
    audit screen is that there is exactly one, and a claim nobody counts stops being true quietly.
    """
    assert mutating_routes() == ["POST /api/case/{claim_id}/approval"]


def test_read_only_refuses_an_approval_even_with_the_correct_token(
    monkeypatch: pytest.MonkeyPatch, priced_claim: str
) -> None:
    """The order of the locks. Read-only is a property of the deployment; a token is a credential.

    Checking the token first would make a correct token sufficient in a read-only deployment, which
    promotes the one of the two controls that can leak.
    """
    reconfigure(monkeypatch, WCR_READ_ONLY="true", WCR_APPROVER_TOKEN=SENTINEL_TOKEN)
    response = client(crashes_become_500=True).post(
        f"/api/case/{priced_claim}/approval",
        json={"case_version": 1},
        headers={console.APPROVER_HEADER_NAME: SENTINEL_TOKEN},
    )
    assert response.status_code == 403
    assert response.json()["error"] == "read_only"


def test_read_only_refuses_before_it_reveals_whether_a_claim_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """403 and not 404. A route that has just refused a caller owes them no census of the corpus."""
    reconfigure(monkeypatch, WCR_READ_ONLY="true", WCR_APPROVER_TOKEN=SENTINEL_TOKEN)
    response = client(crashes_become_500=True).post(
        "/api/case/no-such-claim/approval",
        json={"case_version": 1},
        headers={console.APPROVER_HEADER_NAME: SENTINEL_TOKEN},
    )
    assert response.status_code == 403


def test_an_instance_with_no_configured_token_can_approve_nothing(
    monkeypatch: pytest.MonkeyPatch, priced_claim: str
) -> None:
    """`WCR_APPROVER_TOKEN` has no default, so a deployment that forgot it fails closed."""
    reconfigure(monkeypatch, WCR_READ_ONLY="false")
    response = client(crashes_become_500=True).post(
        f"/api/case/{priced_claim}/approval", json={"case_version": 1}
    )
    assert response.status_code == 403
    assert response.json()["error"] == "no_approver_configured"


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="absent"),
        pytest.param({console.APPROVER_HEADER_NAME: ""}, id="empty"),
        pytest.param({console.APPROVER_HEADER_NAME: "not-the-token"}, id="wrong"),
        pytest.param({console.APPROVER_HEADER_NAME: SENTINEL_TOKEN + "x"}, id="prefix"),
        # A raw header byte above 127, sent as bytes because an HTTP header is bytes on the wire and
        # the client refuses to encode a non-ASCII `str` into one. Starlette decodes what arrives as
        # latin-1, so the handler sees a non-ASCII `str`, and `hmac.compare_digest` raises
        # `TypeError` on one of those. Before the comparison was moved onto encoded bytes this
        # request produced a 500 from the one mutating route, which means a caller holding no
        # credential at all could make the console report itself as broken.
        pytest.param({console.APPROVER_HEADER_NAME: b"tok\xe9n"}, id="non-ascii"),
    ],
)
def test_a_wrong_or_absent_token_is_refused_with_403_and_never_a_crash(
    monkeypatch: pytest.MonkeyPatch, priced_claim: str, headers: dict[str, str | bytes]
) -> None:
    reconfigure(monkeypatch, WCR_READ_ONLY="false", WCR_APPROVER_TOKEN=SENTINEL_TOKEN)
    response = client(crashes_become_500=True).post(
        f"/api/case/{priced_claim}/approval", json={"case_version": 1}, headers=headers
    )
    assert response.status_code == 403, f"expected a refusal, got {response.status_code}"
    assert response.json()["error"] == "approver_token_rejected"


def test_an_approval_binds_the_case_and_the_case_version(
    monkeypatch: pytest.MonkeyPatch, priced_claim: str
) -> None:
    """ "Approved" and "approved *this*" are different claims, and only the second is recorded.

    The event is what kill condition D is graded from elsewhere; what is asserted here is the
    binding — a case identifier and a case version on the same record — and that the response does
    not echo the credential that produced it.
    """
    reconfigure(monkeypatch, WCR_READ_ONLY="false", WCR_APPROVER_TOKEN=SENTINEL_TOKEN)
    before = len(console._console_log)
    response = client(crashes_become_500=True).post(
        f"/api/case/{priced_claim}/approval",
        json={"case_version": 7, "note": "invoice checked against the commissioning record"},
        headers={console.APPROVER_HEADER_NAME: SENTINEL_TOKEN},
    )
    assert response.status_code == 201
    recorded = response.json()["recorded"]
    assert recorded["case_id"] == priced_claim
    assert recorded["case_version"] == 7
    assert recorded["kind"] == APPROVAL_GRANTED
    assert len(console._console_log) == before + 1
    assert SENTINEL_TOKEN.encode() not in response.content

    # The event is a date, never a clock reading: an audit record stamped with the hour it was
    # granted would make every artifact that quoted it unreproducible.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", recorded["at"])
    assert recorded["event_id"] in client().get("/audit").text


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="no-case-version"),
        pytest.param({"note": "approved"}, id="note-only"),
        pytest.param({"case_version": -1}, id="negative-version"),
        pytest.param({"case_version": 1, "approved_by": "someone"}, id="unknown-field"),
    ],
)
def test_an_approval_body_that_names_no_version_or_an_unknown_field_is_refused(
    monkeypatch: pytest.MonkeyPatch, priced_claim: str, body: dict[str, Any]
) -> None:
    """`case_version` is required and `extra="forbid"` does real work.

    A default version would produce an approval bound to a version nobody stated. A tolerated extra
    field would let `approved_by` be sent, ignored, and believed — which is how an approval comes to
    record a person who never saw the case.
    """
    reconfigure(monkeypatch, WCR_READ_ONLY="false", WCR_APPROVER_TOKEN=SENTINEL_TOKEN)
    response = client(crashes_become_500=True).post(
        f"/api/case/{priced_claim}/approval",
        json=body,
        headers={console.APPROVER_HEADER_NAME: SENTINEL_TOKEN},
    )
    assert response.status_code == 422


def test_an_unknown_claim_is_404_once_every_lock_has_been_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconfigure(monkeypatch, WCR_READ_ONLY="false", WCR_APPROVER_TOKEN=SENTINEL_TOKEN)
    response = client(crashes_become_500=True).post(
        "/api/case/no-such-claim/approval",
        json={"case_version": 1},
        headers={console.APPROVER_HEADER_NAME: SENTINEL_TOKEN},
    )
    assert response.status_code == 404


def test_the_audit_screen_names_the_header_and_the_variable_but_no_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader is told how to authenticate, never with what."""
    reconfigure(monkeypatch, WCR_APPROVER_TOKEN=SENTINEL_TOKEN)
    body = client().get("/audit").text
    assert console.APPROVER_HEADER_NAME in body
    assert "WCR_APPROVER_TOKEN" in body
    assert SENTINEL_TOKEN not in body
