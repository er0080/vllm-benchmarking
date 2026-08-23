"""What `/server_info` says, read from payloads a real engine produced.

Both fixtures came off vLLM 0.25.1 on a dual-3090 host: one engine speculating with ngram at
depth 3, one not speculating at all, everything else equal. A hand-written fixture would
encode what the author believed the shape was, which is the belief under test.

The distinction these tests exist to protect is three-way, not two-way. "Speculating",
"the engine says it is not", and "nobody asked the engine" are three different facts about a
run, and the third is the state of every run recorded before protocol 7. Collapsing it into
the second would let a chart claim a comparison it has no evidence for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllmbench_protocol.server_info import NO_SPECULATION, parse_speculation

FIXTURES = Path(__file__).parent / "fixtures"


def payload(name: str) -> dict:
    return json.loads((FIXTURES / f"server_info_vllm_0_25_1_{name}.json").read_text())


class TestARealSpeculatingEngine:
    def test_method_and_depth_are_read(self) -> None:
        found = parse_speculation(payload("speculative"))
        assert found is not None
        assert (found.method, found.tokens) == ("ngram", 3)
        assert found.is_speculating

    def test_the_depth_matches_what_the_engine_was_asked_for(self) -> None:
        """Captured from `--speculative-config '{"method":"ngram","num_speculative_tokens":3}'`.

        Pinning the number as well as the field guards against reading a neighbouring key
        that happens to be an int — `prompt_lookup_max` is 4 in the same object.
        """
        found = parse_speculation(payload("speculative"))
        assert found is not None and found.tokens == 3


class TestARealNonSpeculatingEngine:
    def test_the_key_is_present_and_null(self) -> None:
        """The fact the whole three-state design rests on. If vLLM ever omits the key
        instead, this fails and the parser correctly starts answering "unknown"."""
        assert payload("no_speculation")["vllm_config"]["speculative_config"] is None

    def test_it_reads_as_an_answer_not_a_silence(self) -> None:
        found = parse_speculation(payload("no_speculation"))
        assert found is not None
        assert (found.method, found.tokens) == (NO_SPECULATION, 0)
        assert not found.is_speculating


class TestSilenceIsNeverAnAnswer:
    @pytest.mark.parametrize(
        "unusable",
        [
            None,
            {},
            {"vllm_config": None},
            {"vllm_config": {}},
            {"vllm_config": "a text-format dump"},
            "not json at all",
            [],
        ],
        ids=["none", "empty", "null-config", "no-key", "text-format", "string", "list"],
    )
    def test_an_unreadable_payload_is_unknown(self, unusable: object) -> None:
        assert parse_speculation(unusable) is None

    def test_the_text_format_is_not_silently_accepted(self) -> None:
        """`/server_info` defaults to `config_format=text`, which returns a repr string.

        Getting that back means somebody dropped the query parameter. It must read as
        "could not tell", not as "not speculating" — the second would mark every run on
        that host as non-speculative, including the ones that were.
        """
        assert parse_speculation({"vllm_config": "VllmConfig(model='x', ...)"}) is None

    @pytest.mark.parametrize(
        "broken",
        [
            {"method": None, "num_speculative_tokens": 3},
            {"method": "", "num_speculative_tokens": 3},
            {"method": "ngram"},
            {"method": "ngram", "num_speculative_tokens": None},
            {"method": "ngram", "num_speculative_tokens": 0},
            {"method": "ngram", "num_speculative_tokens": -1},
            {"method": "ngram", "num_speculative_tokens": "3"},
            {"method": "ngram", "num_speculative_tokens": True},
        ],
        ids=[
            "no-method",
            "empty-method",
            "no-depth",
            "null-depth",
            "zero-depth",
            "negative-depth",
            "string-depth",
            "bool-depth",
        ],
    )
    def test_a_configured_but_undescribable_speculation_is_unknown(self, broken: dict) -> None:
        """Speculation is on but cannot be labelled. Inventing a label is worse than NULL:
        the label is what charts group by, so an invented one groups wrongly and silently."""
        assert parse_speculation({"vllm_config": {"speculative_config": broken}}) is None


class TestTheMethodIsNotGuessedFromTheModelField:
    def test_a_method_absent_from_the_object_is_not_taken_from_model(self) -> None:
        """The captured ngram payload carries both `model: "ngram"` and `method: "ngram"`,
        so a parser reading the wrong one passes on that fixture. It must not fall back."""
        assert (
            parse_speculation({"vllm_config": {"speculative_config": {"model": "ngram"}}}) is None
        )

    def test_a_draft_model_method_survives_verbatim(self) -> None:
        """`method` and `model` differ for a draft-model speculator, which is the case that
        makes reading `model` produce a nonsense series label like a HF repo id."""
        found = parse_speculation(
            {
                "vllm_config": {
                    "speculative_config": {
                        "method": "eagle3",
                        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                        "num_speculative_tokens": 5,
                    }
                }
            }
        )
        assert found is not None
        assert (found.method, found.tokens) == ("eagle3", 5)


class TestTheFixturesAreWhatTheyClaim:
    def test_the_two_fixtures_differ_only_in_speculation(self) -> None:
        """Otherwise the speculative fixture proves less than it looks like it proves."""
        spec = payload("speculative")["vllm_config"]
        plain = payload("no_speculation")["vllm_config"]
        assert spec["model_config"]["model"] == plain["model_config"]["model"]

    def test_neither_fixture_carries_a_token(self) -> None:
        """`ModelConfig.hf_token` is `bool | str | None` and is dumped verbatim, so this
        endpoint can leak one. The captured host sets none — asserted, because a fixture
        that silently gained a secret would be committed the same way this one was."""
        for name in ("speculative", "no_speculation"):
            assert payload(name)["vllm_config"]["model_config"]["hf_token"] is None
