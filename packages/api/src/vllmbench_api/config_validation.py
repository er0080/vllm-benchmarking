"""Checking a candidate vLLM configuration before it costs a model load.

The point of this is arithmetic on time. A bad config is discovered when `vllm serve`
exits — which on a large model is several minutes into a sweep that has already claimed
the host, and if it is the *second* config block, after the first has already run. The
checks here cost microseconds and catch the mistakes that produce that outcome.

**Reading, not rewriting.** Invariant 5 says validate, do not transform. Nothing here
emits YAML, reorders keys, fills defaults, or hands back a "corrected" config. Every
finding names a key and says what is wrong with it; what to do about it stays with the
author, and what gets stored stays the exact bytes that reach `vllm serve --config`.

**The catalogue is captured, never authored.** Which arguments exist and what values they
take comes from a real vLLM's argparse (:mod:`vllmbench_protocol.serve_args`). A
hand-written list would encode a belief about vLLM's option set, and a validator that
believes in an argument vLLM has dropped will pass a config that cannot start — at the
one moment somebody is trusting it not to.

**Warnings are not weak errors.** An error means the engine will refuse this file. A
warning means it will start and probably not do what the author meant. Collapsing them
would make the difference between "this cannot run" and "check you meant this"
invisible, and the second kind is the one that quietly produces a valid-looking result.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any

import yaml

from vllmbench_protocol.serve_args import ArgumentSpec, ServeArguments

#: Type names from the captured catalogue that this module knows how to check. Anything
#: else — `parse_dataclass`, `loads`, a lambda — is left alone: a value of a type we
#: cannot verify is not evidence of a mistake, and inventing a rule for it would produce
#: false errors on configs that run.
CHECKABLE = {"int", "float", "str", "bool", "human_readable_int", "human_readable_int_or_auto"}

#: `max-model-len: 8k` is valid. The parser accepts a plain integer or a human-readable
#: suffix, and rejecting the suffixed form would fail a config vLLM runs happily.
_HUMAN_INT = re.compile(r"^\d+(\.\d+)?[kKmMgG]?$")


class Severity(enum.StrEnum):
    """Whether the engine will refuse this, or merely surprise you."""

    #: `vllm serve` will not start with this config.
    ERROR = "error"
    #: It will start. Whether it does what was meant is another question.
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing wrong, and where."""

    severity: Severity
    message: str
    #: The configuration key at fault, when there is one. Absent for whole-document
    #: problems like a syntax error.
    key: str | None = None
    #: 1-based, for an editor that wants to put a marker on it. Absent when the problem
    #: is not attributable to a line.
    line: int | None = None
    #: What to do instead, when there is an unambiguous answer. Never a rewritten config
    #: — a suggestion is offered to the author, not applied behind them.
    suggestion: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """What the engine found, and what it checked against."""

    findings: list[Finding] = field(default_factory=list)
    #: The vLLM version whose argument catalogue was used. Stated rather than assumed,
    #: because validating against a different version than the target host runs is a
    #: normal situation here and the reader has to be able to tell.
    checked_against: str = ""
    #: False when the target host runs a version we have no catalogue for, so the checks
    #: came from the reference instead. Not an error — benchmarking one vLLM version
    #: against another is a headline use of this tool — but it changes what a clean
    #: result means.
    exact_version_match: bool = True

    @property
    def valid(self) -> bool:
        """Whether anything here would stop the engine starting."""
        return not any(f.severity is Severity.ERROR for f in self.findings)


class _LineTrackingLoader(yaml.SafeLoader):
    """A loader that keeps the line each top-level key appeared on.

    Also refuses duplicate keys instead of silently taking the last, which is plain YAML
    behaviour and a genuinely dangerous one here: a config declaring
    `gpu-memory-utilization` twice runs at whichever value came second, hashes as a
    distinct config from the same file with one of them removed, and gives no sign in
    any result that the first line did nothing.
    """


def _line_of(node: yaml.Node) -> int:
    return int(node.start_mark.line) + 1


def _parse(text: str) -> tuple[dict[str, Any] | None, list[Finding]]:
    """Parse the document, reporting syntax and shape problems as findings.

    Returns ``None`` for the mapping when the document could not be read at all — there
    is nothing further to check, and continuing would report every key as missing.
    """
    try:
        root = yaml.compose(text, Loader=_LineTrackingLoader)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        return None, [
            Finding(
                Severity.ERROR,
                f"This is not valid YAML: {getattr(exc, 'problem', None) or exc}.",
                line=(mark.line + 1) if mark is not None else None,
            )
        ]

    if root is None:
        return None, [Finding(Severity.ERROR, "The configuration is empty.")]

    if not isinstance(root, yaml.MappingNode):
        return None, [
            Finding(
                Severity.ERROR,
                "A vLLM config file is a mapping of settings to values, and this is not "
                "one. Each line should read `key: value`.",
                line=_line_of(root),
            )
        ]

    findings: list[Finding] = []
    values: dict[str, Any] = {}
    lines: dict[str, int] = {}
    seen: dict[str, int] = {}
    constructor = _LineTrackingLoader(text)
    for key_node, value_node in root.value:
        if not isinstance(key_node, yaml.ScalarNode):
            findings.append(
                Finding(
                    Severity.ERROR, "Setting names must be plain text.", line=_line_of(key_node)
                )
            )
            continue
        key = str(key_node.value)
        if key in seen:
            findings.append(
                Finding(
                    Severity.ERROR,
                    f"`{key}` is set twice, on lines {seen[key]} and {_line_of(key_node)}. "
                    "YAML keeps only the last one, so the earlier line has no effect and "
                    "nothing downstream will say so.",
                    key=key,
                    line=_line_of(key_node),
                )
            )
            continue
        seen[key] = _line_of(key_node)
        lines[key] = _line_of(value_node)
        values[key] = constructor.construct_object(value_node, deep=True)

    # Line numbers ride along on the mapping rather than in a parallel structure, so a
    # caller that only wants the values can ignore them.
    values["__lines__"] = lines
    return values, findings


def _type_finding(key: str, value: Any, spec: ArgumentSpec, line: int | None) -> Finding | None:
    """Check one value against the type the parser will hand it to."""
    if spec.is_flag or spec.type == "bool":
        if not isinstance(value, bool):
            return Finding(
                Severity.ERROR,
                f"`{key}` is an on/off switch and needs `true` or `false`, not {value!r}.",
                key=key,
                line=line,
                suggestion=f"{key}: true",
            )
        return None

    if spec.type == "int":
        # bool is an int in Python and emphatically not one here.
        if isinstance(value, bool) or not isinstance(value, int):
            return Finding(
                Severity.ERROR,
                f"`{key}` must be a whole number, and {value!r} is not one.",
                key=key,
                line=line,
            )
        return None

    if spec.type == "float":
        if isinstance(value, bool) or not isinstance(value, int | float):
            return Finding(
                Severity.ERROR,
                f"`{key}` must be a number, and {value!r} is not one.",
                key=key,
                line=line,
            )
        return None

    if spec.type in ("human_readable_int", "human_readable_int_or_auto"):
        if spec.type.endswith("or_auto") and value == "auto":
            return None
        if isinstance(value, bool) or not (
            isinstance(value, int) or (isinstance(value, str) and _HUMAN_INT.match(value))
        ):
            allowed = "a whole number, or a size like `8k`"
            if spec.type.endswith("or_auto"):
                allowed += ", or `auto`"
            return Finding(
                Severity.ERROR,
                f"`{key}` must be {allowed}; {value!r} is none of those.",
                key=key,
                line=line,
            )
        return None

    return None


def _choice_finding(key: str, value: Any, spec: ArgumentSpec, line: int | None) -> Finding | None:
    """Check a value against the parser's own list of accepted ones."""
    if not spec.choices or isinstance(value, bool) or value is None:
        return None
    text = str(value)
    if text in spec.choices:
        return None
    # A looser cutoff than the key namespace gets, deliberately. Choices are a small
    # closed set where a near miss is almost always the intended value, and the mistake
    # this exists for — `fp16` for `float16` — sits at 0.55.
    near = get_close_matches(text, spec.choices, n=1, cutoff=0.5)
    listed = ", ".join(f"`{choice}`" for choice in spec.choices)
    return Finding(
        Severity.ERROR,
        f"`{key}` does not accept {value!r}. It takes one of: {listed}.",
        key=key,
        line=line,
        suggestion=f"{key}: {near[0]}" if near else None,
    )


def validate_config(
    yaml_text: str,
    arguments: ServeArguments,
    *,
    gpu_count: int | None = None,
    exact_version_match: bool = True,
    tensor_parallel_is_swept: bool = False,
) -> ValidationResult:
    """Check a candidate configuration against what one vLLM version accepts.

    ``gpu_count`` enables the checks that need to know the machine — a topology asking
    for more devices than exist is the cheapest possible thing to catch and one of the
    most expensive to discover from a failed sweep.

    ``tensor_parallel_is_swept`` says the sweep planner will be rewriting
    ``tensor-parallel-size`` on this config, which turns whatever the author wrote there
    into something that will be ignored. That is worth saying out loud; it is not an
    error, because writing a baseline value and then sweeping over it is reasonable.
    """
    values, findings = _parse(yaml_text)
    if values is None:
        return ValidationResult(
            findings=findings,
            checked_against=arguments.vllm_version,
            exact_version_match=exact_version_match,
        )

    lines: dict[str, int] = values.pop("__lines__", {})

    for key, value in values.items():
        line = lines.get(key)
        found = arguments.spec_for(key)
        if found is None:
            near = get_close_matches(
                key.replace("_", "-"), list(arguments.arguments), n=1, cutoff=0.7
            )
            findings.append(
                Finding(
                    Severity.ERROR,
                    f"vLLM {arguments.vllm_version} has no `{key}` setting.",
                    key=key,
                    line=line,
                    suggestion=f"{near[0]}: {value}" if near else None,
                )
            )
            continue

        _, spec = found
        if spec.type in CHECKABLE:
            if finding := _type_finding(key, value, spec, line):
                findings.append(finding)
                continue
        if finding := _choice_finding(key, value, spec, line):
            findings.append(finding)
            continue

    findings.extend(_topology_findings(values, lines, gpu_count, tensor_parallel_is_swept))

    if "model" not in values and "model" not in {k.replace("_", "-") for k in values}:
        findings.append(
            Finding(
                Severity.WARNING,
                "No `model` is set. vLLM will need one from somewhere — the command line, "
                "or a default — and a config that does not name its model cannot be "
                "handed to another host and expected to do the same thing.",
                key="model",
            )
        )

    # Errors first, then by position, so the thing that stops it running is at the top.
    findings.sort(key=lambda f: (f.severity is not Severity.ERROR, f.line or 0))
    return ValidationResult(
        findings=findings,
        checked_against=arguments.vllm_version,
        exact_version_match=exact_version_match,
    )


def _topology_findings(
    values: dict[str, Any],
    lines: dict[str, int],
    gpu_count: int | None,
    tensor_parallel_is_swept: bool,
) -> list[Finding]:
    """Checks that need to know about the machine, or about the sweep.

    Kept apart from the per-key loop because these are the only rules here with an
    opinion about what the settings *mean* together, and that opinion is ours rather than
    something read out of vLLM's parser.
    """
    findings: list[Finding] = []

    def whole_number(name: str) -> int | None:
        for key, value in values.items():
            if (
                key.replace("_", "-") == name
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                return value
        return None

    tensor = whole_number("tensor-parallel-size")
    pipeline = whole_number("pipeline-parallel-size")
    line = lines.get("tensor-parallel-size") or lines.get("tensor_parallel_size")

    if tensor is not None and tensor < 1:
        findings.append(
            Finding(
                Severity.ERROR,
                f"`tensor-parallel-size` must be at least 1, not {tensor}.",
                key="tensor-parallel-size",
                line=line,
            )
        )
    if gpu_count is not None and tensor is not None:
        needed = tensor * (pipeline or 1)
        if needed > gpu_count:
            topology = (
                f"tensor-parallel-size {tensor} by pipeline-parallel-size {pipeline}"
                if pipeline
                else f"tensor-parallel-size {tensor}"
            )
            findings.append(
                Finding(
                    Severity.ERROR,
                    f"This asks for {needed} GPU(s) ({topology}) and the target host has "
                    f"{gpu_count}. vLLM will fail during startup.",
                    key="tensor-parallel-size",
                    line=line,
                )
            )

    if tensor_parallel_is_swept and tensor is not None:
        findings.append(
            Finding(
                Severity.WARNING,
                f"The sweep sets `tensor-parallel-size` on each point, so the {tensor} "
                "written here will be replaced and never reaches the engine.",
                key="tensor-parallel-size",
                line=line,
            )
        )

    return findings
