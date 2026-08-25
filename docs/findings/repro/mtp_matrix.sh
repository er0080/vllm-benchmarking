#!/bin/bash
# Overnight matrix: can the push custom all-reduce serve MTP, and is it correct and fast?
#
# What is already known going in:
#   * MTP + NCCL peer-to-peer transport = hang. PHB and SYS fail identically (2/8 requests,
#     7 tokens, a 300s engine timeout); only leaving NCCL_P2P_LEVEL unset works. So it is
#     P2P being enabled at all, not the level chosen.
#   * Every MTP arm so far ran with disable-custom-all-reduce: true, so the push kernel has
#     never actually been on an MTP path.
#
# The bet: custom all-reduce replaces NCCL for exactly the small messages MTP's drafter
# leans on, so push-on/NCCL-P2P-off may be both correct and faster than either.
#
# Correctness is captured per arm, not inferred. Throughput without a token check is the
# trap this whole investigation exists to document -- the NaN arm reported +7.7%.
set -uo pipefail

BASE=$HOME/p2p-ab
SUMMARY=$BASE/mtp-matrix-summary.txt
PORT=8905
DS=/home/eric/vllmbench-datasets/notebook-edit-00f3871d786b.json

cd "$HOME/vllm-env" && . .venv/bin/activate

export HF_HOME="$HOME/hf"
export PATH="$PATH:/usr/local/cuda-13.3/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda-13.3/lib64"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_SERVER_DEV_MODE=1

PID=""
cleanup() {
  [ -n "$PID" ] && kill -TERM "$PID" 2>/dev/null
  for _ in $(seq 1 40); do kill -0 "${PID:-0}" 2>/dev/null || break; sleep 1; done
  kill -9 "${PID:-0}" 2>/dev/null
  # By pattern is unsafe over ssh (it matches the caller); these are our own children.
  for w in $(pgrep -P "${PID:-0}" 2>/dev/null); do kill -9 "$w" 2>/dev/null; done
  sleep 5
}
trap cleanup EXIT INT TERM

run_arm() {
  local label=$1 cfg=$2 nccl=$3 push=$4
  local log=$BASE/mx-$label-engine.log
  local bench=$BASE/mx-$label-bench.log
  local greedy=$BASE/mx-$label-greedy.json

  if [ "$nccl" = "unset" ]; then unset NCCL_P2P_LEVEL; else export NCCL_P2P_LEVEL=$nccl; fi
  if [ "$push" = "on" ]; then
    export VLLM_CUSTOM_ALLREDUCE_PUSH=1
    export VLLM_PUSHAR_LIB=$BASE/../p2p-probes/pushar/pushar.so
  else
    unset VLLM_CUSTOM_ALLREDUCE_PUSH VLLM_PUSHAR_LIB
  fi

  echo "=== $label | cfg=$(basename "$cfg") nccl=$nccl push=$push ==="
  vllm serve --config "$cfg" --port $PORT > "$log" 2>&1 &
  PID=$!

  local ready=""
  for _ in $(seq 1 180); do
    curl -sf --max-time 2 "localhost:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 $PID 2>/dev/null || break
    sleep 5
  done
  if [ -z "$ready" ]; then
    echo "RESULT $label: ENGINE_NEVER_READY" | tee -a "$SUMMARY"
    cleanup; PID=""; return
  fi

  # Correctness first: a hang during greedy capture is itself the answer, and if the
  # engine survives we get a hash to compare against the known-good NCCL reference.
  timeout 600 python3 "$BASE/greedy_capture.py" "$greedy" "http://localhost:$PORT" \
    > "$BASE/mx-$label-greedy.log" 2>&1
  local ghash
  ghash=$(python3 -c "
import json,sys
try:
    print(json.load(open('$greedy'))['combined_sha256'][:16])
except Exception:
    print('NONE')
" 2>/dev/null)

  # Only benchmark if the engine is still alive; otherwise the numbers are a timeout.
  local ok bad gen dur tps
  if kill -0 $PID 2>/dev/null; then
    timeout 900 vllm bench serve --backend vllm --base-url "http://localhost:$PORT" \
      --model Qwen/Qwen3.8-27B-FP8 --served-model-name Qwen3.8-27B \
      --dataset-name sharegpt --dataset-path "$DS" \
      --num-prompts 8 --max-concurrency 2 --request-rate inf \
      --sharegpt-output-len 512 \
      --save-result --result-filename "$BASE/mx-$label-result.json" \
      > "$bench" 2>&1
    ok=$(grep -aoP "Successful requests:\s+\K[0-9]+" "$bench" | head -1)
    bad=$(grep -aoP "Failed requests:\s+\K[0-9]+" "$bench" | head -1)
    gen=$(grep -aoP "Total generated tokens:\s+\K[0-9]+" "$bench" | head -1)
    dur=$(grep -aoP "Benchmark duration \(s\):\s+\K[0-9.]+" "$bench" | head -1)
    tps=$(grep -aoP "Output token throughput \(tok/s\):\s+\K[0-9.]+" "$bench" | head -1)
  else
    ok=- ; bad=- ; gen=- ; dur=- ; tps=-
  fi

  local stalls dead selftest pushon
  stalls=$(grep -c "No available shared memory broadcast block" "$log")
  dead=$(grep -c "EngineDeadError" "$log")
  selftest=$(grep -c "failed a numerical self-test" "$log")
  pushon=$(grep -ci "pushar\|push.based\|push kernel" "$log")

  echo "RESULT $label: ok=${ok:-?} failed=${bad:-?} gen=${gen:-?} dur=${dur:-?}s tok_s=${tps:-?} greedy=$ghash stalls=$stalls dead=$dead selftest_disabled=$selftest push_mentions=$pushon" \
    | tee -a "$SUMMARY"

  cleanup
  PID=""
}

: > "$SUMMARY"
echo "matrix started $(date -u +%FT%TZ)" >> "$SUMMARY"

#        label            config                    NCCL      push
run_arm  ref-nccl-nop2p   "$BASE/mtp-repro.yaml"    unset     off
run_arm  push-nop2p       "$BASE/mtp-pushar.yaml"   unset     on
run_arm  push-sys         "$BASE/mtp-pushar.yaml"   SYS       on
run_arm  push-nop2p-nospec "$BASE/pushar-nospec.yaml" unset   on

echo "matrix finished $(date -u +%FT%TZ)" >> "$SUMMARY"
