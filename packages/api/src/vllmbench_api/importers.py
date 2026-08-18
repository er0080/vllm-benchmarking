"""Reading a `vllm bench sweep serve` output directory.

Written against a **captured** directory, not the documentation — `tests/fixtures/
vllm_sweep_v0.25.1/` is a real one, produced by running the tool. Two things that capture
revealed and that no amount of reading would have:

* `summary.json` is exactly the list of the `run=N.json` payloads. Same keys, no
  aggregation. It is a convenience file, not a second contract.
* **Nothing in the output carries provenance.** No vLLM version, no GPU, no host, no
  device count. See `docs/adr/0003-importing-upstream-sweeps.md`; the consequence is that
  the operator has to supply it and we have to record that they did.

This module only *parses*. It creates nothing and knows nothing about the database, so
every judgement it makes about a directory name can be tested against strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: `SERVE--max_num_seqs=4-BENCH--max_concurrency=2-num_prompts=8`
#:
#: Split on the literal markers rather than parsed as one pattern: the separator between
#: two parameters and the separator inside a value are both `-`, so a general pattern
#: cannot tell `num_prompts=8` from a value containing a dash. The markers are the only
#: unambiguous landmarks in the name.
_SERVE_MARKER = "SERVE--"
_BENCH_MARKER = "-BENCH--"

_PARAM = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_.]*)=(?P<value>.*)$")

#: A file this importer will read. `summary.json` is skipped deliberately — it duplicates
#: the run files, and the run files carry `run_number`, which is what makes replicates
#: distinguishable and the spread computable.
RUN_FILE = re.compile(r"^run=(?P<index>\d+)\.json$")


class SweepDirectoryError(ValueError):
    """The directory is not one this importer recognises."""


def _coerce(text: str) -> Any:
    """Turn a parameter value back into the type it was written from.

    The directory name is the only record of these, and it is a string. `max_num_seqs=4`
    has to come back as an int or the reconstructed config claims the engine was given
    the string "4".
    """
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_params(segment: str) -> dict[str, Any]:
    """`max_concurrency=2-num_prompts=8` into a mapping.

    Split on `-` followed by something that looks like `key=`, so a value containing a
    dash — a model id, a path — survives. A segment that does not parse is returned as a
    single opaque entry rather than dropped: losing a parameter silently would make two
    different sweep points look like the same one.
    """
    if not segment:
        return {}
    parts = re.split(r"-(?=[A-Za-z_][A-Za-z0-9_.]*=)", segment)
    params: dict[str, Any] = {}
    for part in parts:
        match = _PARAM.match(part)
        if match is None:
            params[part] = True
            continue
        params[match["key"]] = _coerce(match["value"])
    return params


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One directory: a server configuration measured under one benchmark setting."""

    directory: str
    serve_params: dict[str, Any]
    bench_params: dict[str, Any]
    #: One entry per `run=N.json`, ordered by N — the replicate index the tool assigned.
    runs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def replicates(self) -> int:
        return len(self.runs)


def parse_directory_name(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover the serve and bench parameters encoded in a point's directory name.

    That name is the *only* place the serve overrides appear in the directory layout, so
    a name this cannot read is a point whose configuration is unknown — which is refused
    rather than imported as a nameless config.
    """
    if not name.startswith(_SERVE_MARKER):
        raise SweepDirectoryError(
            f"{name!r} is not a sweep point directory: expected a name beginning "
            f"{_SERVE_MARKER!r}, as `vllm bench sweep serve` writes."
        )
    body = name[len(_SERVE_MARKER) :]
    if _BENCH_MARKER not in body:
        raise SweepDirectoryError(
            f"{name!r} has no {_BENCH_MARKER!r} section, so its benchmark settings "
            "cannot be recovered."
        )
    serve_text, _, bench_text = body.partition(_BENCH_MARKER)
    return parse_params(serve_text), parse_params(bench_text)


def build_points(files: dict[str, Any]) -> list[SweepPoint]:
    """Group parsed JSON files, keyed by path relative to the experiment directory.

    Paths rather than a directory handle because the control plane never has the
    operator's filesystem — invariant 1 keeps this service away from the machine under
    test, and an upload is the only thing it can see.

    Values are typed loosely because the directory legitimately contains more than one
    shape: `summary.json` is a JSON *array*, and rejecting it at the door would refuse a
    perfectly good directory for containing a file this importer does not need.
    """
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for path, payload in files.items():
        parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
        if len(parts) < 2:
            continue
        directory, filename = parts[-2], parts[-1]
        match = RUN_FILE.match(filename)
        if match is None:
            # summary.json and anything else the tool may add later.
            continue
        if not isinstance(payload, dict):
            raise SweepDirectoryError(
                f"{path} is not a benchmark result object. A `run=N.json` file holds one "
                "result; a list here usually means summary.json was renamed."
            )
        grouped.setdefault(directory, {})[int(match["index"])] = payload

    points: list[SweepPoint] = []
    for directory in sorted(grouped):
        serve_params, bench_params = parse_directory_name(directory)
        runs = [grouped[directory][index] for index in sorted(grouped[directory])]
        points.append(
            SweepPoint(
                directory=directory,
                serve_params=serve_params,
                bench_params=bench_params,
                runs=runs,
            )
        )
    if not points:
        raise SweepDirectoryError(
            "no `run=N.json` files were found. Point this at the experiment directory — "
            "the one containing the SERVE--…-BENCH--… folders — rather than at the "
            "results root or a single point."
        )
    return points


def reconstructed_yaml(serve_params: dict[str, Any], model: str | None) -> str:
    """The server configuration, as far as it can be recovered.

    **This is not the file that ran, and it does not claim to be.** The server was
    configured by command-line flags; only the parameters the operator chose to sweep
    appear in the output at all, so anything fixed in `--serve-cmd` is unrecoverable.
    Invariant 5 is about what gets passed to `vllm serve --config`, and this was not —
    which is why the text says so in a comment that travels with it.
    """
    lines = [
        "# Reconstructed by import from a `vllm bench sweep serve` output directory.",
        "# These are the swept parameters only: anything fixed in --serve-cmd was not",
        "# recorded by the tool and is therefore absent. Not a runnable configuration.",
    ]
    if model:
        lines.append(f"model: {model}")
    for key in sorted(serve_params):
        value = serve_params[key]
        rendered = (
            "null" if value is None else str(value).lower() if isinstance(value, bool) else value
        )
        lines.append(f"{key.replace('_', '-')}: {rendered}")
    return "\n".join(lines) + "\n"
