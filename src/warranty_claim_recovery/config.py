"""Configuration, read from the environment once, with the two dangerous values defaulting to off.

`approver_token` and `llm_api_key` have **no defaults**. That is the whole design: a deployment
that forgot the approver token can approve nothing, and one that forgot the model key raises rather
than quietly composing with something else. Both fail closed, and a failure that is closed is a
failure someone notices.

`read_only` defaults to **true** for the same reason. A misconfigured public instance that refuses
every mutation is an inconvenience; one that accepts them is an incident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

__all__ = [
    "APPROVER_TOKEN_ENV",
    "DATABASE_URL_ENV",
    "DEFAULT_DATABASE_URL",
    "DEFAULT_REDIS_URL",
    "LLM_API_KEY_ENV",
    "REDIS_URL_ENV",
    "Settings",
    "get_settings",
]

DATABASE_URL_ENV: Final = "WCR_DATABASE_URL"
REDIS_URL_ENV: Final = "WCR_REDIS_URL"
# The NAME of an environment variable, not a value. S105 pattern-matches the identifier.
APPROVER_TOKEN_ENV: Final = "WCR_APPROVER_TOKEN"  # noqa: S105
LLM_API_KEY_ENV: Final = "WCR_LLM_API_KEY"

#: The compose file's ports, chosen so they cannot collide with the other repositories in this
#: workspace or with a locally installed PostgreSQL or Redis.
DEFAULT_DATABASE_URL: Final = (
    "postgresql+psycopg://warranty:warranty_local_only@127.0.0.1:15441/warranty_claim_recovery"
)
DEFAULT_REDIS_URL: Final = "redis://127.0.0.1:16380/0"


def _flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean; use true or false")


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    #: No default. Without it the HITL gate refuses every approval.
    approver_token: str | None
    #: No default. Without it the abstractive arm raises.
    llm_api_key: str | None
    read_only: bool
    corpus_dir: str
    artifacts_dir: str
    embedding_cache_dir: str
    environment: str

    @property
    def live_model_available(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def approvals_possible(self) -> bool:
        return bool(self.approver_token)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        database_url=os.environ.get(DATABASE_URL_ENV, DEFAULT_DATABASE_URL),
        redis_url=os.environ.get(REDIS_URL_ENV, DEFAULT_REDIS_URL),
        approver_token=os.environ.get(APPROVER_TOKEN_ENV) or None,
        llm_api_key=os.environ.get(LLM_API_KEY_ENV) or None,
        read_only=_flag("WCR_READ_ONLY", default=True),
        corpus_dir=os.environ.get("WCR_CORPUS_DIR", "data/generated"),
        artifacts_dir=os.environ.get("WCR_ARTIFACTS_DIR", "artifacts"),
        embedding_cache_dir=os.environ.get("WCR_EMBEDDING_CACHE", ".fastembed_cache"),
        environment=os.environ.get("WCR_ENVIRONMENT", "local"),
    )
