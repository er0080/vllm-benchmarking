#!/bin/bash
# One MTP benchmark under a named environment variation. Vary one thing at a time.
#
# Usage: mtp_arm.sh <label> [VAR_TO_UNSET ...]
set -uo pipefail

LABEL=${1:?usage: mtp_arm.sh <label> [vars to unset...]}
shift

export HF_HOME="$HOME/hf"
export PATH="$PATH:/usr/local/cuda-13.3/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda-13.3/lib64"
NCCL_P2P_LEVEL_SET=${NCCL_P2P_LEVEL_SET:-SYS}; export NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL_SET
export VLLM_CUSTOM_ALLREDUCE_PUSH=1
export VLLM_PUSHAR_LIB=/home/eric/p2p-probes/pushar/pushar.so
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_SERVER_DEV_MODE=1
[ -n "${NCCL_PROTO_SET:-}" ] && export NCCL_PROTO=$NCCL_PROTO_SET

for v in "$@"; do unset "$v"; echo "unset $v"; done

LOG=$HOME/p2p-ab/mtp-$LABEL-engine.log
BENCH=$HOME/p2p-ab/mtp-$LABEL-bench.log
PORT=8903
CFG=${CFG_SET:-$HOME/p2p-ab/mtp-repro.yaml}
DS=/home/eric/vllmbench-datasets/notebook-edit-00f3871d786b.json

cd "$HOME/vllm-env" && . .venv/bin/activate

cleanup() {
  [ -n "${PID:-}" ] && kill -TERM "$PID" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "${PID:-0}" 2>/dev/null || break; sleep 1; done
  kill -9 "${PID:-0}" 2>/dev/null
  pkill -9 -f "VLLM::Worker" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "--- effective env ---"
for v in NCCL_P2P_LEVEL NCCL_PROTO VLLM_CUSTOM_ALLREDUCE_PUSH VLLM_PUSHAR_LIB PYTORCH_CUDA_ALLOC_CONF; do
  printf "  %-30s %s\n" "$v" "${!v:-<unset>}"
done

vllm serve --config "$CFG" --port $PORT > "$LOG" 2>&1 &
PID=$!
for _ in $(seq 1 150); do
  curl -sf --max-time 2 "localhost:$PORT/health" >/dev/null 2>&1 && { echo "ready"; break; }
  kill -0 $PID 2>/dev/null || { echo "RESULT $LABEL: engine died during load"; exit 1; }
  sleep 5
done

vllm bench serve --backend vllm --base-url "http://localhost:$PORT" \
  --model Qwen/Qwen3.8-27B-FP8 --served-model-name Qwen3.8-27B \
  --dataset-name sharegpt --dataset-path "$DS" \
  --num-prompts 8 --max-concurrency 2 --request-rate inf \
  --sharegpt-output-len 512 \
  --save-result --result-filename "$HOME/p2p-ab/mtp-$LABEL-result.json" \
  > "$BENCH" 2>&1

ok=$(grep -aoP "Successful requests:\s+\K[0-9]+" "$BENCH" | head -1)
bad=$(grep -aoP "Failed requests:\s+\K[0-9]+" "$BENCH" | head -1)
gen=$(grep -aoP "Total generated tokens:\s+\K[0-9]+" "$BENCH" | head -1)
dur=$(grep -aoP "Benchmark duration \(s\):\s+\K[0-9.]+" "$BENCH" | head -1)
tps=$(grep -aoP "Output token throughput \(tok/s\):\s+\K[0-9.]+" "$BENCH" | head -1)
hang=$(grep -c "No available shared memory broadcast block" "$LOG")
dead=$(grep -c "EngineDeadError" "$LOG")

echo "RESULT $LABEL: ok=${ok:-?} failed=${bad:-?} generated=${gen:-?} duration=${dur:-?}s out_tok_s=${tps:-?} shm_stalls=$hang enginedead=$dead"
