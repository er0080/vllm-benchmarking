"""Compare the shipped vLLM argument catalogue against one dumped from a live vLLM.

Run by tier 2 against the CPU backend container. A stale catalogue is worse than no
catalogue: the validation engine would reject a setting the pinned vLLM accepts, or pass
one it has dropped, at the moment somebody is trusting it not to.

    docker exec -i vllm-cpu python - < scripts/capture_serve_args.py > raw.json
    python scripts/check_serve_args_catalogue.py raw.json <shipped.json>

A script rather than inline YAML so it can be run by hand against any vLLM image, and so
the comparison is something a reader can step through rather than a quoted shell string.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any


def load_dump(path: pathlib.Path) -> dict[str, Any]:
    """Read a capture, tolerating vLLM's startup logging.

    vLLM writes an INFO line to stdout before the dumper prints, and suppressing that
    would mean importing vLLM differently from the way the agent does. Keeping from the
    first brace is the smaller compromise.
    """
    raw = path.read_text()
    start = raw.find("{")
    if start < 0:
        raise SystemExit(f"::error::{path} contains no JSON object")
    return json.loads(raw[start:])


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <captured.json> <shipped.json>")

    live = load_dump(pathlib.Path(sys.argv[1]))
    shipped = json.loads(pathlib.Path(sys.argv[2]).read_text())

    if live == shipped:
        print(f"catalogue matches: {len(live['arguments'])} arguments")
        return 0

    live_args, shipped_args = live["arguments"], shipped["arguments"]
    differences = {
        "added": sorted(set(live_args) - set(shipped_args)),
        "removed": sorted(set(shipped_args) - set(live_args)),
        "changed": sorted(
            key for key in set(live_args) & set(shipped_args) if live_args[key] != shipped_args[key]
        ),
    }
    if live["vllm_version"] != shipped["vllm_version"]:
        print(
            f"::error::catalogue is for vLLM {shipped['vllm_version']} but the running "
            f"server is {live['vllm_version']}"
        )

    print("::error::the shipped vllm serve argument catalogue is out of date")
    for label, keys in differences.items():
        if keys:
            print(f"{label}: {', '.join(keys)}")
    print("Regenerate with scripts/capture_serve_args.py and commit the result.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
