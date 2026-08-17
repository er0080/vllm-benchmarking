#!/usr/bin/env python3
"""Assert that an installed agent environment stands on its own.

The agent is installed into someone else's virtualenv — the one vLLM lives in, on a host
this repository never sees again. Whatever ends up in that environment has to keep
working with no source tree anywhere on the machine.

That is not automatic. ``vllmbench-protocol`` is a workspace member, so from inside a
checkout ``uv pip install ./packages/agent`` resolves it through ``tool.uv.sources`` and
installs it *editable*: a ``.pth`` file pointing back at the checkout. Nothing warns, the
agent runs, and it keeps running right up until the clone is deleted — at which point it
dies with ``ModuleNotFoundError: vllmbench_protocol`` on a box with no source and no clue.
That happened on the first real GPU host.

So this checks the property that actually matters, which is not "does it import" but
"does it still import once the thing it was installed from is gone".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED_PACKAGES = {"vllmbench_agent", "vllmbench_protocol"}


def site_packages(venv: Path) -> Path:
    candidates = sorted(venv.glob("lib/python*/site-packages"))
    if not candidates:
        # Windows layout, and a clearer failure than an IndexError.
        candidates = sorted(venv.glob("Lib/site-packages"))
    if not candidates:
        sys.exit(f"no site-packages under {venv}")
    return candidates[-1]


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit("usage: check_agent_install.py <venv-path>")

    site = site_packages(Path(sys.argv[1]))
    problems: list[str] = []

    # An editable install leaves a .pth naming the directory it depends on. Report the
    # path it points at, because that is the whole diagnosis.
    for pth in sorted(site.glob("_editable_impl_*.pth")):
        problems.append(f"{pth.name} makes this install depend on {pth.read_text().strip()}")

    seen: set[str] = set()
    for dist_info in sorted(site.glob("vllmbench_*.dist-info")):
        name = dist_info.name.split("-")[0]
        seen.add(name)

        direct_url = dist_info / "direct_url.json"
        if not direct_url.is_file():
            # Installed from an index rather than a path or VCS. Fine, and self-contained.
            continue

        record = json.loads(direct_url.read_text())
        if record.get("dir_info", {}).get("editable"):
            problems.append(f"{name} is installed editable from {record.get('url')}")
        elif "dir_info" in record:
            # Non-editable, but still built from a local directory. The files were
            # copied, so it survives deletion — worth naming, not worth failing.
            print(f"note: {name} was built from a local directory ({record.get('url')})")
        elif "vcs_info" in record:
            commit = record["vcs_info"].get("commit_id", "?")
            print(f"ok: {name} from {record.get('url')} @ {commit[:12]}")
        else:
            print(f"ok: {name} from {record.get('url')}")

    for missing in sorted(EXPECTED_PACKAGES - seen):
        problems.append(f"{missing} is not installed at all")

    if problems:
        for problem in problems:
            print(f"::error::{problem}")
        print(
            "\nThe agent must not reference a source checkout. Install it from git:\n"
            '  uv pip install "git+https://github.com/er0080/vllm-benchmarking'
            '@main#subdirectory=packages/agent"'
        )
        return 1

    print(f"{len(seen)} packages, none referencing a source checkout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
