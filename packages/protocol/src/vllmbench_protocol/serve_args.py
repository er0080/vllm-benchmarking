"""What ``vllm serve`` actually accepts.

The catalogue is **captured from a real parser, never written by hand**. `scripts/`
holds the dumper; it reads `argparse` inside a real vLLM rather than parsing `--help`,
because the help text is prose that reflows between releases while the parser is the
thing that accepts or rejects a config.

This is the same rule that governs `bench_result` and `metrics`, applied to a third
upstream surface. A hand-written catalogue would encode the author's belief about what
vLLM takes, which is exactly the belief under test — and the failure mode is worse here
than a wrong field name, because a validator that believes in an argument vLLM dropped
will pass a config that cannot start, at the moment somebody is trusting it not to.

The bundled capture is the *reference* version. A GPU host running something else is a
supported and expected situation (see the version policy in CLAUDE.md): validation
against the reference is still useful, and says which version it used.
"""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import BaseModel, Field

DATA = Path(__file__).parent / "data"


class ArgumentSpec(BaseModel):
    """One accepted argument, as the parser describes it."""

    #: The parser's type callable, by name. ``int``, ``float``, ``str`` and ``bool`` are
    #: checkable; the rest — ``parse_dataclass``, ``loads`` and friends — are recorded
    #: verbatim so a value of an unrecognised type is left alone rather than guessed at.
    type: str
    #: Present only where the parser restricts the value. This is the check that catches
    #: `dtype: fp16`, which is a name vLLM does not have (it wants `float16` or `half`).
    choices: list[str] | None = None
    #: ``0`` marks a store_true/store_false flag, which in YAML is a boolean.
    nargs: int | str | None = None
    #: Other spellings the parser answers to, including the ``no-`` form of a flag.
    aliases: list[str] = Field(default_factory=list)

    @property
    def is_flag(self) -> bool:
        return self.nargs == 0


class ServeArguments(BaseModel):
    """Every argument one vLLM version accepts, and which version that was."""

    vllm_version: str
    arguments: dict[str, ArgumentSpec]

    def spec_for(self, key: str) -> tuple[str, ArgumentSpec] | None:
        """Look up a key as a config might spell it.

        vLLM's parser normalizes underscores to dashes, so `tensor_parallel_size` and
        `tensor-parallel-size` are the same argument and a validator that only knew one
        of them would reject a config the engine accepts.
        """
        canonical = key.replace("_", "-")
        spec = self.arguments.get(canonical)
        if spec is not None:
            return canonical, spec
        for name, candidate in self.arguments.items():
            if canonical in candidate.aliases:
                return name, candidate
        return None


@functools.lru_cache(maxsize=8)
def load_serve_arguments(vllm_version: str) -> ServeArguments | None:
    """The captured catalogue for one version, or ``None`` if we do not have it."""
    path = DATA / f"vllm_serve_args_v{vllm_version}.json"
    if not path.is_file():
        return None
    return ServeArguments.model_validate_json(path.read_text())


def available_versions() -> list[str]:
    """Versions with a captured catalogue, newest last."""
    prefix, suffix = "vllm_serve_args_v", ".json"
    return sorted(p.name[len(prefix) : -len(suffix)] for p in DATA.glob(f"{prefix}*{suffix}"))


def reference_serve_arguments() -> ServeArguments:
    """The catalogue for the pinned reference version.

    Raises rather than returning ``None``: the reference capture is shipped with this
    package, so its absence is a broken build rather than a configuration a caller
    should handle.
    """
    versions = available_versions()
    if not versions:  # pragma: no cover - would mean the wheel shipped without its data
        raise RuntimeError(f"no vllm serve argument catalogue found in {DATA}")
    loaded = load_serve_arguments(versions[-1])
    assert loaded is not None
    return loaded
