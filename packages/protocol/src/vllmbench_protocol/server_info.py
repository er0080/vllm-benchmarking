"""The vLLM ``/server_info`` contract, for what the engine resolved rather than what was asked.

Captured from a live vLLM 0.25.1 on a dual-3090 host, both ways: speculating (ngram, depth
3) and not. The payloads are ``tests/fixtures/server_info_vllm_0_25_1_{speculative,
no_speculation}.json``. Written against those, never against the documentation — see
CLAUDE.md, "Upstream contracts are verified, never assumed".

Three things the real payloads settle:

**The endpoint is gated.** ``/server_info`` is attached by ``register_vllm_dev_api_routers``,
which ``api_server.py`` calls only under ``VLLM_SERVER_DEV_MODE``. Measured: 404 without it,
200 with. The agent sets that variable already, because the ``/reset_*_cache`` endpoints live
behind the same gate and upstream's own sweep runner sets it for exactly that reason.

**Off is a value, not an absence.** With no speculation the key is *present* and ``null``::

    {"vllm_config": {"speculative_config": null, ...}}

So three states are distinguishable and stay distinguishable: the engine is speculating, the
engine says it is not, and nobody asked the engine. Collapsing the last two is the same
mistake as reading a NULL ``environment_status`` as a clean environment.

**The payload can contain a HuggingFace token.** ``ModelConfig.hf_token`` is typed
``bool | str | None`` and is dumped verbatim; on the captured host it is null only because
none is set. The whole response also carries 222 environment variables and the host's pip
list. Nothing here returns the payload, logs it, or stores it — two scalars are lifted out
and the rest is dropped on the floor. Keep it that way.

The parallel with :mod:`vllmbench_protocol.metrics` is deliberate: what the engine *is* comes
from the engine, exactly as what the engine *did* does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["NO_SPECULATION", "Speculation", "parse_speculation"]

#: What ``speculative_method`` holds when the engine states it is not speculating. A word
#: rather than NULL, because NULL already means "the engine did not say" and a run that was
#: measured without speculation is a different fact from a run nobody asked about.
NO_SPECULATION = "none"

#: Where the resolved speculative settings live in the response.
_CONFIG_KEY = "vllm_config"
_SPECULATIVE_KEY = "speculative_config"
#: vLLM normalizes `--speculative-config '{"method": ...}'` onto this field, and fills it in
#: even when the caller only gave `model` — the captured ngram payload has both `"model":
#: "ngram"` and `"method": "ngram"`. Reading `method` is reading what the engine settled on.
_METHOD_FIELD = "method"
_TOKENS_FIELD = "num_speculative_tokens"


@dataclass(frozen=True, slots=True)
class Speculation:
    """What one engine resolved for speculative decoding.

    ``method`` is :data:`NO_SPECULATION` when the engine reported that it is not
    speculating. There is no instance of this class meaning "unknown" — that is the absence
    of an instance, which callers represent as ``None`` and the database as NULL.
    """

    method: str
    tokens: int

    @property
    def is_speculating(self) -> bool:
        return self.method != NO_SPECULATION


def parse_speculation(payload: Any) -> Speculation | None:
    """Read the resolved speculative configuration out of a ``/server_info`` response.

    Returns ``None`` when the payload cannot answer — a response from a version that shapes
    this differently, or anything that is not the object we captured. That is deliberately
    not the same as :data:`NO_SPECULATION`: a run measured against an engine we could not
    ask must not claim the engine denied it.
    """
    if not isinstance(payload, dict):
        return None
    config = payload.get(_CONFIG_KEY)
    if not isinstance(config, dict) or _SPECULATIVE_KEY not in config:
        return None

    speculative = config[_SPECULATIVE_KEY]
    if speculative is None:
        return Speculation(method=NO_SPECULATION, tokens=0)
    if not isinstance(speculative, dict):
        return None

    method = speculative.get(_METHOD_FIELD)
    tokens = speculative.get(_TOKENS_FIELD)
    if not isinstance(method, str) or not method:
        # Speculation is configured but unnamed. Refusing to answer beats inventing a
        # label: the label is what charts group by, and an invented one groups wrongly.
        return None
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        return None
    return Speculation(method=method, tokens=tokens)
