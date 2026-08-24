"""What the agent records about the environment it launched an engine in.

The gap this closes is not hypothetical. `config_hash` is the hash of the config text, so
it cannot see anything set in the environment — and on this project's own GPU host,
`NCCL_P2P_LEVEL=SYS` moved per-GPU output throughput 13.4% at concurrency 16 with the
config byte-identical. Worse, `disable-custom-all-reduce: false` hashes the same whether
the engine then runs a kernel that sums correctly or one that returns NaN for every
element, because an environment variable decides which.

So the tests that matter here are about *coverage* and *fidelity*: that a variable nobody
anticipated is still captured, and that what is recorded is what the child actually got.
"""

from __future__ import annotations

from vllmbench_agent.hardware import (
    ENGINE_ENV_REDACTED,
    engine_environment,
)


class TestWhatIsCaptured:
    def test_the_variables_that_forced_this_field(self) -> None:
        """The two settings that changed a measurement without changing a config."""
        captured = engine_environment(
            {
                "NCCL_P2P_LEVEL": "SYS",
                "VLLM_CUSTOM_ALLREDUCE_PUSH": "1",
                "VLLM_PUSHAR_LIB": "/home/eric/p2p-probes/pushar/pushar.so",
            }
        )
        assert captured == {
            "NCCL_P2P_LEVEL": "SYS",
            "VLLM_CUSTOM_ALLREDUCE_PUSH": "1",
            "VLLM_PUSHAR_LIB": "/home/eric/p2p-probes/pushar/pushar.so",
        }

    def test_an_unanticipated_vllm_variable_is_captured(self) -> None:
        """The whole reason for prefix matching.

        `VLLM_CUSTOM_ALLREDUCE_PUSH` did not exist when this column was designed. An
        enumerated allowlist would have recorded nothing about it and the runs would have
        been indistinguishable from NCCL runs.
        """
        captured = engine_environment({"VLLM_SOMETHING_INVENTED_NEXT_QUARTER": "7"})
        assert captured == {"VLLM_SOMETHING_INVENTED_NEXT_QUARTER": "7"}

    def test_loader_state_is_captured(self) -> None:
        """Which NCCL gets bound is decided here, and it is not prefix-shaped."""
        captured = engine_environment({"LD_LIBRARY_PATH": "/opt/nccl/lib", "LD_PRELOAD": "x.so"})
        assert captured == {"LD_LIBRARY_PATH": "/opt/nccl/lib", "LD_PRELOAD": "x.so"}

    def test_unrelated_environment_is_left_out(self) -> None:
        captured = engine_environment({"HOME": "/home/eric", "SHELL": "/bin/zsh", "TERM": "xterm"})
        assert captured == {}

    def test_empty_environment_reports_empty_not_none(self) -> None:
        """`{}` and NULL mean different things downstream and must not collapse.

        `{}` is an agent stating none of these were set. NULL is a run from before the
        agent could answer. Reading the first as the second would let silence pass for an
        observation.
        """
        assert engine_environment({}) == {}

    def test_the_agents_own_settings_are_not_engine_settings(self) -> None:
        """`VLLMBENCH_` must not match `VLLM_`; the underscore is load-bearing.

        This is the first of two defences on the shared token. The second is redaction,
        checked below. Neither is relied on alone.
        """
        captured = engine_environment(
            {"VLLMBENCH_TOKEN": "s3cret", "VLLMBENCH_LOG_LEVEL": "INFO", "VLLM_LOGGING_LEVEL": "D"}
        )
        assert captured == {"VLLM_LOGGING_LEVEL": "D"}


class TestSecrets:
    def test_a_secret_keeps_its_name_and_loses_its_value(self) -> None:
        """Presence is provenance; the value is not.

        Whether the engine required an API key changes how it behaved and belongs on the
        run. What the key was must not reach a database or a JSON API.
        """
        captured = engine_environment({"VLLM_API_KEY": "sk-live-abcdef"})
        assert captured == {"VLLM_API_KEY": ENGINE_ENV_REDACTED}

    def test_every_secret_marker_redacts(self) -> None:
        captured = engine_environment(
            {
                "VLLM_AUTH_TOKEN": "t",
                "VLLM_API_KEY": "k",
                "VLLM_CLIENT_SECRET": "s",
                "VLLM_DB_PASSWORD": "p",
                "VLLM_AWS_CREDENTIAL": "c",
            }
        )
        assert set(captured.values()) == {ENGINE_ENV_REDACTED}
        assert len(captured) == 5

    def test_redaction_is_case_insensitive_on_the_name(self) -> None:
        captured = engine_environment({"VLLM_hf_token": "hf_xxx"})
        assert captured == {"VLLM_hf_token": ENGINE_ENV_REDACTED}

    def test_a_harmless_name_keeps_its_value(self) -> None:
        """Redaction must not be so eager that the field stops being useful."""
        captured = engine_environment({"VLLM_PUSHAR_LIB": "/opt/pushar.so"})
        assert captured == {"VLLM_PUSHAR_LIB": "/opt/pushar.so"}


class TestStability:
    def test_result_is_sorted_so_equal_settings_compare_equal(self) -> None:
        """Two runs launched the same way must produce equal mappings.

        Parent environments do not preserve order, and a comparison that reports a
        difference because a dict was built in another order is a false positive in
        exactly the place this field is read.
        """
        one = engine_environment({"VLLM_B": "2", "NCCL_A": "1"})
        two = engine_environment({"NCCL_A": "1", "VLLM_B": "2"})
        assert list(one) == list(two) == ["NCCL_A", "VLLM_B"]
        assert one == two

    def test_an_empty_string_is_a_value_not_an_absence(self) -> None:
        """`NCCL_P2P_LEVEL=` is not the same instruction as leaving it unset."""
        captured = engine_environment({"NCCL_P2P_LEVEL": ""})
        assert captured == {"NCCL_P2P_LEVEL": ""}
        assert captured != {}
