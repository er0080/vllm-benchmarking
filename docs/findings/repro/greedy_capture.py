"""Capture greedy completions from a running engine, for comparison across all-reduce paths.

Gate 2 of the peer-to-peer A/B. A driver that moves tensors between devices incorrectly
produces a model that is fast, plausible and wrong, and a throughput benchmark reports
tokens per second either way. So before any figure from the patched arm is trusted, the
same prompts are run against both all-reduce paths and the tokens are diffed.

Uses `/v1/completions`, not `/v1/chat/completions`, deliberately. This model is served with
`reasoning-parser: qwen3` and `preserve_thinking`, so the chat endpoint splits output into
`reasoning_content` and `content` and returns a null `content` whenever the token budget is
spent before the model stops thinking — which is what broke the first attempt. The raw
completions endpoint applies no chat template and no reasoning parser, so what comes back
is the model's own token stream, which is the thing under test. Serving-layer formatting is
not.

Sent one at a time at temperature 0. Batching changes floating-point reduction order for
reasons that have nothing to do with the driver, and the point is to isolate the all-reduce
path — vLLM's custom kernel against the NCCL path it replaces.

Usage:  python greedy_capture.py <out.json> [base_url]
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request

PROMPTS = [
    "The first ten prime numbers are:",
    "def reverse_linked_list(head):",
    "PCIe peer-to-peer DMA reduces all-reduce latency because",
    "17 multiplied by 243 equals",
    "The seven continents, in alphabetical order, are:",
    "Translate to French: The measurement is more important than the feature.",
    "Tensor parallelism differs from pipeline parallelism in that",
    "A haiku about a graphics card at full load:",
]

MAX_TOKENS = 256


def complete(base_url: str, model: str, prompt: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1234,
            "max_tokens": MAX_TOKENS,
        }
    ).encode()
    req = urllib.request.Request(  # noqa: S310 - fixed loopback URL
        base_url + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
        payload = json.load(resp)

    choice = payload["choices"][0]
    text = choice.get("text")
    if not text:
        # Never silently hash an empty string: two empty captures would compare equal and
        # report a pass for a check that measured nothing.
        raise SystemExit(
            f"engine returned no text for {prompt!r} "
            f"(finish_reason={choice.get('finish_reason')!r}); "
            "the comparison would be vacuous, so refusing to record it"
        )
    return text


def main() -> int:
    out_path = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"

    with urllib.request.urlopen(base_url + "/v1/models", timeout=30) as resp:  # noqa: S310
        model = json.load(resp)["data"][0]["id"]

    results = []
    for prompt in PROMPTS:
        text = complete(base_url, model, prompt)
        results.append(
            {
                "prompt": prompt,
                "chars": len(text),
                "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                "text": text,
            }
        )

    report = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "endpoint": "/v1/completions",
        "combined_sha256": hashlib.sha256(
            "".join(r["text"] for r in results).encode()
        ).hexdigest(),
        "completions": results,
    }
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"  model:    {model}")
    print(f"  combined: {report['combined_sha256']}")
    for r in results:
        print(f"    {r['sha256']}  {r['chars']:>4} chars  {r['prompt'][:48]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
