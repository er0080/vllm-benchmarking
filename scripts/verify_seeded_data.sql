-- Assert that the rows seeded at the previous schema survived this branch's migrations,
-- with their values unchanged.
--
-- Row counts are not enough. A migration that rewrote a throughput figure, or dropped a
-- device's telemetry while keeping its peer, or lost the raw payload, would pass a
-- count-based check and be exactly the silent corruption this project exists to prevent.
-- So every seeded value is compared, not merely counted.
--
-- Fails by raising, so a non-zero psql exit is the signal. Each message names the thing
-- that changed, because "verification failed" sends the reader to read a migration
-- diff from scratch.

DO $$
DECLARE
    expected CONSTANT uuid := '44444444-4444-4444-4444-444444444444';
    v_status text;
    v_throughput double precision;
    v_per_gpu double precision;
    v_ttft double precision;
    v_raw jsonb;
    v_extra jsonb;
    v_engine int;
    v_devices int;
    v_environment text;
    v_speculative text;
    v_dataset text;
    v_engine_env jsonb;
BEGIN
    -- The run, and its provenance.
    SELECT status INTO v_status FROM run WHERE id = expected;
    IF v_status IS DISTINCT FROM 'succeeded' THEN
        RAISE EXCEPTION 'the seeded run is % after migrating, expected succeeded', v_status;
    END IF;

    SELECT raw_result INTO v_raw FROM run WHERE id = expected;
    IF v_raw IS DISTINCT FROM '{"completed": 100, "duration": 60.0}'::jsonb THEN
        RAISE EXCEPTION
            'the raw benchmark payload changed: %. It is the only way to recompute a '
            'mis-flattened result, so it must survive verbatim', v_raw;
    END IF;

    -- The measurement.
    SELECT output_token_throughput_tok_sec, output_token_throughput_per_gpu, ttft_ms_p99, extra
    INTO v_throughput, v_per_gpu, v_ttft, v_extra
    FROM run_summary WHERE run_id = expected;

    IF v_throughput IS DISTINCT FROM 1234.5 THEN
        RAISE EXCEPTION 'aggregate throughput changed from 1234.5 to %', v_throughput;
    END IF;
    IF v_per_gpu IS DISTINCT FROM 617.25 THEN
        RAISE EXCEPTION
            'per-GPU throughput changed from 617.25 to %. Invariant 8: this is the '
            'number comparison views default to', v_per_gpu;
    END IF;
    IF v_ttft IS DISTINCT FROM 99.9 THEN
        RAISE EXCEPTION 'p99 TTFT changed from 99.9 to %', v_ttft;
    END IF;
    IF v_extra IS DISTINCT FROM '{"kept": "verbatim"}'::jsonb THEN
        RAISE EXCEPTION 'the unmapped-fields column changed: %', v_extra;
    END IF;

    -- Telemetry, per device. Counting the total would miss a migration that kept one
    -- device and dropped its peer, which is precisely the imbalance signal that makes a
    -- tensor-parallel run diagnosable.
    SELECT count(*) INTO v_engine FROM engine_sample WHERE run_id = expected;
    IF v_engine <> 2 THEN
        RAISE EXCEPTION 'engine samples went from 2 to %', v_engine;
    END IF;

    SELECT count(DISTINCT gpu_index) INTO v_devices FROM gpu_sample WHERE run_id = expected;
    IF v_devices <> 2 THEN
        RAISE EXCEPTION
            'gpu_sample now covers % device(s), expected 2. A host-level aggregate '
            'destroys the imbalance signal', v_devices;
    END IF;

    -- A run measured before the environment check existed must stay NULL, not acquire a
    -- default. NULL reads back as "the agent did not say"; any non-null value here would
    -- be this migration inventing a claim about a machine nobody checked.
    SELECT environment_status INTO v_environment FROM run WHERE id = expected;
    IF v_environment IS NOT NULL THEN
        RAISE EXCEPTION
            'a historical run acquired environment_status %, which asserts something '
            'about a host that was never checked', v_environment;
    END IF;

    -- The same argument, for the same reason, one release later. A run measured before
    -- the framework asked the engine about speculation has no answer, and NULL is that
    -- answer. Defaulting it to 'none' would file every historical run in the
    -- non-speculative arm of a comparison it was never part of — and unlike a wrong
    -- number, nothing about it would look wrong.
    SELECT speculative_method INTO v_speculative FROM run WHERE id = expected;
    IF v_speculative IS NOT NULL THEN
        RAISE EXCEPTION
            'a historical run acquired speculative_method %, which claims the engine was '
            'asked something it was never asked', v_speculative;
    END IF;

    -- Likewise the dataset. This column has existed since the initial schema and was
    -- NULL on every row until protocol 7 filled it in going forward; a backfill would be
    -- inventing a corpus identity for data nobody hashed.
    SELECT dataset_identity INTO v_dataset FROM run WHERE id = expected;
    IF v_dataset IS NOT NULL THEN
        RAISE EXCEPTION
            'a historical run acquired dataset_identity %, which describes bytes nobody '
            'read', v_dataset;
    END IF;

    -- And the engine environment. The distinction this column has to keep is between
    -- NULL and '{}': NULL is a run from before the agent could report what it launched
    -- the engine with, '{}' is an agent saying none of those settings were set. A
    -- server-side default of '{}' would turn every historical run into the second, which
    -- reads as an observation and is not one. That matters here more than most: the
    -- settings this column holds decide which all-reduce kernel ran.
    SELECT engine_env INTO v_engine_env FROM run WHERE id = expected;
    IF v_engine_env IS NOT NULL THEN
        RAISE EXCEPTION
            'a historical run acquired engine_env %, which claims the agent reported an '
            'environment it had no way to send', v_engine_env;
    END IF;

    RAISE NOTICE 'seeded data survived the migration with every value intact';
END
$$;
