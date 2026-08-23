"""What data a benchmark actually read, computed where the data lives.

Invariant 6 requires every run to record its dataset identity, and invariant 1 puts the
data out of the control plane's reach: ``--dataset-path`` names a file on the GPU host, and
nothing on the other side of the boundary can see it. So the agent answers.

The point is narrow and worth stating exactly. A workload is content-addressed on the
*arguments* — dataset name, path, subset, extra flags — which pins what was asked for. It
cannot pin what was there. Overwrite the file at that path and every run before and after
claims the identical workload, groups into one series, and attributes the difference to
whichever config changed. Nothing errors. This module is what makes those two runs say
different things about themselves.

Four forms, each self-describing, because a bare hash cannot say what it hashed:

``sha256:<digest>:<bytes>``
    A local file, hashed whole. The ordinary case.

``sha256-head-tail:<digest>:<bytes>``
    A local file past :data:`FULL_HASH_MAX_BYTES`, digested from its first and last
    :data:`SAMPLE_BYTES` plus its length. Weaker, and *labelled* weaker: an edit in the
    middle of a 40 GB file will not move it. Reading forty gigabytes before every
    benchmark would be a real cost on the machine under test, and this is the trade —
    made visible rather than hidden behind a prefix that claims a full hash.

``hf:<repo>@<revision>``
    A HuggingFace dataset, with the commit the local cache actually resolved to. The repo
    id alone is a moving target.

``generated:<name>:<knobs>``
    A generator rather than a corpus — ``random``, ``prefix_repetition``. Deterministic
    given its knobs and vLLM's seed, so the knobs *are* the identity. Recorded explicitly
    so that "no dataset identity" stops being ambiguous between "generated" and "nobody
    looked".

Deliberately not "synthetic": invariant 7 uses that word for runs that did not measure real
hardware, and a generated prompt set measured on a real GPU is not one of those. Reusing the
word would put the quarantine vocabulary on runs that do not belong in quarantine.

Returning ``None`` means the agent could not tell. That is a fifth state and stays distinct
from all four above, on the same reasoning as :data:`vllmbench_protocol.EnvironmentStatus`:
silence must never read as an answer.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from vllmbench_protocol.wire import BenchRequest

log = logging.getLogger(__name__)

__all__ = ["identify_dataset"]

#: Files up to this size are hashed in full. ShareGPT's full corpus is ~650 MB, so the
#: ordinary case stays well inside it and takes a second or two of cold-cache read.
FULL_HASH_MAX_BYTES = 2 * 1024**3

#: How much of each end of an oversized file goes into the sampled digest.
SAMPLE_BYTES = 64 * 1024 * 1024

_READ_CHUNK = 1024 * 1024

#: Dataset names that generate their prompts rather than reading them. From vLLM 0.25.1's
#: own `--dataset-name` choices; anything unrecognised with no path falls through to None
#: rather than being guessed at.
_GENERATED = frozenset({"random", "random-mm", "random-rerank", "prefix_repetition"})

#: The generator knobs this agent can set. `--seed` is not among them: vLLM defaults it to
#: 0 and we never pass it, so it is a constant rather than a variable and putting it in the
#: string would imply a control we do not exercise.
_GENERATED_KNOBS = ("num_prompts", "random_input_len", "random_output_len")


def identify_dataset(request: BenchRequest) -> str | None:
    """Describe the data this request will be measured against.

    Never raises. A run must not fail because its provenance was awkward to compute — the
    measurement is the valuable thing and an unidentified dataset is recorded as
    unidentified, which is exactly what the NULL is for.
    """
    try:
        return _identify(request)
    except OSError as exc:
        # Unreadable, vanished, a permissions problem. `vllm bench serve` is about to hit
        # the same wall and fail the run with a better message than we could write here.
        log.warning("could not identify dataset %r: %s", request.dataset_path, exc)
        return None


def _identify(request: BenchRequest) -> str | None:
    path = request.dataset_path

    # A local file wins over everything else, including `--hf-name`: upstream's help says
    # to set `--hf-name` *when* `--dataset-path` is a local path, so the two together mean
    # a local file carrying a HuggingFace dataset's schema. The bytes are what was read.
    if path and Path(path).is_file():
        return _identify_file(Path(path))

    # With no local file, `--dataset-path` is the HF repo id (upstream: "Path to the
    # sharegpt/sonnet dataset or the HF dataset ID if using HF dataset").
    repo = request.hf_name or (path if request.dataset_name == "hf" else None)
    if repo:
        return f"hf:{repo}@{_hf_revision(repo) or 'unresolved'}"

    if request.dataset_name in _GENERATED:
        knobs = ",".join(
            f"{name}={getattr(request, name)}"
            for name in _GENERATED_KNOBS
            if getattr(request, name) is not None
        )
        return f"generated:{request.dataset_name}:{knobs}"

    # A named dataset with no path and no repo — `sonnet` without its file, say. We know
    # its name and nothing about its contents, and its name is already the workload hash.
    return None


def _identify_file(path: Path) -> str:
    size = path.stat().st_size
    if size <= FULL_HASH_MAX_BYTES:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(_READ_CHUNK):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}:{size}"

    digest = hashlib.sha256()
    # The size goes in first, so two files sharing both ends but differing in length
    # cannot collide — which is otherwise the easy way to defeat a head-and-tail digest.
    digest.update(str(size).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(SAMPLE_BYTES))
        handle.seek(-SAMPLE_BYTES, os.SEEK_END)
        digest.update(handle.read(SAMPLE_BYTES))
    log.info("dataset %s is %d bytes; identity is head-and-tail, not a full hash", path, size)
    return f"sha256-head-tail:{digest.hexdigest()}:{size}"


def _hf_revision(repo: str) -> str | None:
    """The commit a HuggingFace dataset resolved to locally, from the hub cache layout.

    Read off disk rather than asked over the network: the agent runs on the machine under
    test and must not make a benchmark wait on huggingface.co, or fail one because the
    host is offline. The cache is also the honest source — it is what vLLM will load.
    """
    cache = Path(
        os.environ.get("HF_DATASETS_CACHE")
        or os.path.join(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface", "hub")
    )
    ref = cache / f"datasets--{repo.replace('/', '--')}" / "refs" / "main"
    try:
        revision = ref.read_text().strip()
    except OSError:
        return None
    return revision or None
