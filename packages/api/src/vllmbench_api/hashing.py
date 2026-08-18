"""Content addressing for configurations and workloads.

CLAUDE.md: "Two runs claiming the same config must have byte-identical effective
configuration." That is read literally here — the hash covers the exact YAML text, with
only line-ending and trailing-whitespace normalization.

The tempting alternative, parsing the YAML and hashing a canonical form, would make
semantically equivalent configs share a hash. It would also mean *interpreting* the
config, and invariant 5 says validate, do not transform: the moment we parse it we have
opinions about what vLLM options mean, and those opinions rot with every vLLM release.
Two byte-different configs that happen to behave identically getting two hashes is a
minor inefficiency. Two byte-different configs sharing one hash is a corrupted result
set.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def normalize_yaml(text: str) -> str:
    """Normalize only what cannot change meaning: line endings and trailing space."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip() + "\n"


def config_hash(yaml_text: str) -> str:
    return hashlib.sha256(normalize_yaml(yaml_text).encode()).hexdigest()


def workload_hash(fields: dict[str, Any]) -> str:
    """Hash a workload's identity-bearing fields.

    Unlike a config, a workload is our own structure rather than a vLLM artifact, so
    canonicalizing it is safe: sorted keys, no whitespace, nulls preserved. Nulls matter
    — ``max_concurrency: null`` means unbounded, and collapsing it to a default would
    make an uncapped run hash-identical to a capped one.
    """
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _slug(text: str) -> str:
    """A filename-safe version of a config's name.

    Deliberately lossy and deliberately not reversible: this only has to produce a name a
    person can recognise in a downloads folder. The config's identity travels in the hash
    beside it, which is exact.
    """
    kept = [c if c.isalnum() or c in "-_" else "-" for c in text.strip().lower()]
    slug = "".join(kept).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:60] or "config"
