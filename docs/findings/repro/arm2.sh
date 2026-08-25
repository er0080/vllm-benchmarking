#!/bin/bash
# One MTP arm, taking its base environment from ~/.bashrc rather than hardcoding it.
#
# Invoked through `bash -ic`, so ~/.bashrc has already been sourced and its early return for
# non-interactive shells does not apply. That matters now: PYTORCH_CUDA_ALLOC_CONF has been
# removed from it and NCCL settings are being varied per arm, so a script carrying its own
# copy of the environment would silently test yesterday's configuration.
#
# Usage: arm2.sh <label> <config> <nccl_p2p_level|off>
set -uo pipefail

LABEL=${1:?label}
CFG=${2:?config}
NCCL=${3:?nccl level or "off"}

BASE=$HOME/p2p-ab
export VLLM_SERVER_DEV_MODE=1
if [ "$NCCL" = "off" ]; then unset NCCL_P2P_LEVEL; else export NCCL_P2P_LEVEL="$NCCL"; fi

LOG=$BASE/a2-$LABEL-engine.log
BENCH=$BASE/a2-$LABEL-bench.log
GREEDY=$BASE/a2-$LABEL-greedy.json
PORT=8909
DS=/home/eric/vllmbench-datasets/notebook-edit-00f3871d786b.json

cd "$HOME/vllm-env" && . .venv/bin/activate

PID=""
cleanup() {
  [ -n "$PID" ] && kill -TERM "$PID" 2>/dev/null
  for _ in $(seq 1 40); do kill -0 "${PID:-0}" 2>/dev/null || break; sleep 1; done
  kill -9 "${PID:-0}" 2>/dev/null
  for w in $(pgrep -P "${PID:-0}" 2>/dev/null); do kill -9 "$w" 2>/dev/null; done
  sleep 6
}
trap cleanup EXIT INT TERM

echo "=== $LABEL | cfg=$(basename "$CFG") ==="
printf "  %-30s %s\n" \
  NCCL_P2P_LEVEL "${NCCL_P2P_LEVEL:-<unset>}" \
  PYTORCH_CUDA_ALLOC_CONF "${PYTORCH_CUDA_ALLOC_CONF:-<unset>}" \
  VLLM_CUSTOM_ALLREDUCE_PUSH "${VLLM_CUSTOM_ALLREDUCE_PUSH:-<unset>}" \
  gpu-mem-util "$(grep -oP 'gpu-memory-utilization:\s*\K[0-9.]+' "$CFG")" \
  custom-allreduce "$(grep -oP 'disable-custom-all-reduce:\s*\K\S+' "$CFG")"

vllm serve --config "$CFG" --port $PORT > "$LOG" 2>&1 &
PID=$!
ready=""
for _ in $(seq 1 180); do
  curl -sf --max-time 2 "localhost:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 $PID 2>/dev/null || break
  sleep 5
done
[ -n "$ready" ] || { echo "RESULT2 $LABEL: ENGINE_NEVER_READY"; exit 1; }

echo "  free after load: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/')"

timeout 600 python3 "$BASE/greedy_capture.py" "$GREEDY" "http://localhost:$PORT" \
  > "$BASE/a2-$LABEL-greedy.log" 2>&1
ghash=$(python3 -c "
import json
try: print(json.load(open('$GREEDY'))['combined_sha256'][:16])
except Exception: print('NONE')")

ok=- ; bad=- ; gen=- ; dur=- ; tps=-
if kill -0 $PID 2>/dev/null; then
  timeout 900 vllm bench serve --backend vllm --base-url "http://localhost:$PORT" \
    --model Qwen/Qwen3.8-27B-FP8 --served-model-name Qwen3.8-27B \
    --dataset-name sharegpt --dataset-path "$DS" \
    --num-prompts 8 --max-concurrency 2 --request-rate inf --sharegpt-output-len 512 \
    --save-result --result-filename "$BASE/a2-$LABEL-result.json" > "$BENCH" 2>&1
  ok=$(grep -aoP "Successful requests:\s+\K[0-9]+" "$BENCH" | head -1)
  bad=$(grep -aoP "Failed requests:\s+\K[0-9]+" "$BENCH" | head -1)
  gen=$(grep -aoP "Total generated tokens:\s+\K[0-9]+" "$BENCH" | head -1)
  dur=$(grep -aoP "Benchmark duration \(s\):\s+\K[0-9.]+" "$BENCH" | head -1)
  tps=$(grep -aoP "Output token throughput \(tok/s\):\s+\K[0-9.]+" "$BENCH" | head -1)
fi

echo "RESULT2 $LABEL: ok=${ok:-?} failed=${bad:-?} gen=${gen:-?} dur=${dur:-?}s tok_s=${tps:-?} greedy=$ghash stalls=$(grep -c 'No available shared memory broadcast block' "$LOG") dead=$(grep -c EngineDeadError "$LOG")"
