-- Seed a database that is sitting at the *previous* schema, so the next migration has
-- to run against real rows rather than an empty table.
--
-- Forward-only means a migration that cannot run against existing data is a
-- release-blocking bug (CLAUDE.md). An empty-database test cannot find that class of bug
-- at all: a NOT NULL column with no default, a type change that fails on real values, a
-- constraint that no historical row satisfies — every one of those passes against
-- nothing and fails against a month of results.
--
-- Deliberately plain SQL with explicit column lists, and deliberately *not* the ORM.
-- This file is executed against the schema as it exists on `main`, where the current
-- branch's models do not exist yet; an ORM insert would emit the new columns and fail
-- before testing anything. The explicit list is also the point of the file: it is the
-- set of columns whose survival across a migration is being asserted, written down where
-- a reviewer can see it change.
--
-- Only columns present since 0.1.0 are used, so this keeps working as the schema grows.

BEGIN;

INSERT INTO gpu_host (id, name, agent_url, gpu_count, vllm_version, driver_version)
VALUES (
    '11111111-1111-1111-1111-111111111111',
    'seeded-host',
    'http://10.0.0.9:9110',
    2,
    '0.25.1',
    '580.95.05'
);

INSERT INTO server_config (id, config_hash, name, yaml)
VALUES (
    '22222222-2222-2222-2222-222222222222',
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    'seeded-config',
    E'model: facebook/opt-125m\ntensor-parallel-size: 2\n'
);

-- `extra_args` is NOT NULL with a Python-side default, so the ORM fills it and raw SQL
-- must too. Exactly the kind of thing this file exists to keep honest about.
INSERT INTO workload (
    id, workload_hash, name, dataset_name, num_prompts, max_concurrency, extra_args
)
VALUES (
    '33333333-3333-3333-3333-333333333333',
    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    'seeded-workload',
    'random',
    100,
    16,
    '{}'::jsonb
);

-- A terminal run, which is the interesting case: it is immutable by trigger, so a
-- migration that tries to rewrite it in place fails here rather than in production.
-- `replicate_idx` is NOT NULL with a Python-side default too. Both of these are worth
-- the noise: a column whose default lives only in the ORM is invisible to anything that
-- writes rows another way, which is what a migration's backfill is.
INSERT INTO run (
    id, replicate_idx, server_config_id, workload_id, gpu_host_id, status,
    started_at, finished_at, config_hash, workload_hash,
    vllm_version, gpu_model, gpu_count, tensor_parallel_size, pipeline_parallel_size,
    bench_client_location, is_synthetic, initiated_by, raw_result
)
VALUES (
    '44444444-4444-4444-4444-444444444444',
    0,
    '22222222-2222-2222-2222-222222222222',
    '33333333-3333-3333-3333-333333333333',
    '11111111-1111-1111-1111-111111111111',
    'succeeded',
    now() - interval '2 hours',
    now() - interval '1 hour',
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    '0.25.1',
    'NVIDIA GeForce RTX 3090',
    2, 2, 1,
    'loopback',
    false,
    'ui',
    '{"completed": 100, "duration": 60.0}'::jsonb
);

-- The measurement. If a migration can lose this, it can lose a month of results.
INSERT INTO run_summary (
    run_id, successful_requests, failed_requests, benchmark_duration_sec,
    total_input_tokens, total_generated_tokens,
    output_token_throughput_tok_sec, output_token_throughput_per_gpu,
    ttft_ms_median, ttft_ms_p99, tpot_ms_mean, extra
)
VALUES (
    '44444444-4444-4444-4444-444444444444',
    100, 0, 60.0, 12800, 6400,
    1234.5, 617.25,
    42.5, 99.9, 25.0,
    '{"kept": "verbatim"}'::jsonb
);

-- Per-device telemetry, keyed on (run_id, gpu_index, sampled_at). A migration that
-- collapsed that key would destroy the imbalance signal, and it would do so silently.
INSERT INTO engine_sample (run_id, sampled_at, num_requests_running, num_requests_waiting)
VALUES
    ('44444444-4444-4444-4444-444444444444', now() - interval '90 minutes', 16, 0),
    ('44444444-4444-4444-4444-444444444444', now() - interval '89 minutes', 16, 4);

INSERT INTO gpu_sample (run_id, gpu_index, sampled_at, sm_utilization_pct, memory_used_bytes)
VALUES
    ('44444444-4444-4444-4444-444444444444', 0, now() - interval '90 minutes', 95.0, 21000000000),
    ('44444444-4444-4444-4444-444444444444', 1, now() - interval '90 minutes', 61.0, 21000000000);

COMMIT;
