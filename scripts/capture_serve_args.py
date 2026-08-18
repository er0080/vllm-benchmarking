"""Capture ``vllm serve``'s accepted arguments from a real vLLM.

Run inside a container holding the version you want, and write the result into the
protocol package's ``data/`` directory:

    docker run --rm -i --entrypoint python vllm/vllm-openai-cpu:v0.25.1 - \
        < scripts/capture_serve_args.py > /tmp/args.json

vLLM logs to stdout before this prints, so the caller trims everything before the first
``{``. That is deliberate rather than fought: suppressing vLLM's logging would mean
importing it differently from the way the agent does.

Read from ``argparse`` rather than from ``--help``: the help output is prose that
reflows between releases, while the parser is the thing that actually accepts or rejects
a config (CLAUDE.md — upstream contracts are verified, never assumed).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import vllm
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser


def resolve(type_callable: Any) -> Any:
    """Unwrap vLLM's ``_optional_type`` wrapper to the type it actually parses.

    Without this a fifth of the arguments report a type name that says only "this may be
    None" and nothing about what a valid value looks like.
    """
    for _ in range(5):
        if type_callable is None or getattr(type_callable, "__name__", "") != "_optional_type":
            break
        cells = [
            cell.cell_contents
            for cell in (type_callable.__closure__ or ())
            if callable(cell.cell_contents)
        ]
        if not cells:
            break
        type_callable = cells[0]
    return type_callable


def type_name(action: argparse.Action) -> str:
    if isinstance(action, argparse._StoreTrueAction | argparse._StoreFalseAction):
        return "bool"
    resolved = resolve(getattr(action, "type", None))
    if resolved is None:
        return "str"
    return getattr(resolved, "__name__", str(resolved))


def main() -> None:
    parser = make_arg_parser(FlexibleArgumentParser())
    arguments: dict[str, dict[str, Any]] = {}
    for action in parser._actions:
        names = [option for option in action.option_strings if option.startswith("--")]
        if not names:
            continue
        primary = names[0].lstrip("-")
        if primary == "help":
            continue
        choices = list(action.choices) if action.choices else None
        arguments[primary] = {
            "type": type_name(action),
            "choices": sorted(str(choice) for choice in choices) if choices else None,
            "nargs": action.nargs,
            "aliases": sorted({option.lstrip("-") for option in names} - {primary}),
        }

    json.dump(
        {"vllm_version": vllm.__version__, "arguments": arguments},
        sys.stdout,
        indent=2,
        sort_keys=True,
    )


if __name__ == "__main__":
    main()
