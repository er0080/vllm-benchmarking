"""The pinned vLLM reference version.

Read from the ``VLLM_REFERENCE_VERSION`` file at the repo root, or from the environment
variable of the same name when the file is not present (as in an installed wheel).

This is the version CI tier 2 validates against. A GPU host running something else is
warned about and recorded — never blocked. Blocking would prevent comparing vLLM
versions, which is one of the things this tool is for.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path


@functools.cache
def reference_vllm_version() -> str | None:
    from_env = os.environ.get("VLLM_REFERENCE_VERSION")
    if from_env:
        return from_env.strip()

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "VLLM_REFERENCE_VERSION"
        if candidate.is_file():
            for line in candidate.read_text().splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return stripped
    return None
