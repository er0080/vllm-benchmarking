#!/bin/bash
# Gate 2: does the all-reduce path peer access unlocks compute the same tokens?
#
# Compares vLLM's custom all-reduce kernel against the NCCL path it replaces, on the same
# driver, in the same session. That isolates the kernel — comparing across a reboot would
# confound it with everything else a reboot changes.
#
# A driver that moves tensors between devices incorrectly produces a model that is fast,
# plausible and wrong, and a throughput benchmark reports tokens per second either way. So
# no figure from the patched arm is trusted until this passes.
#
# VLLM_SERVER_DEV_MODE is set for the same reason the agent sets it: /server_info and the
# cache-reset endpoints 404 without it. Its absence is what killed the first attempt.
#
# Usage: gate2.sh <label>     e.g. gate2.sh patched
set -euo pipefail

LABEL=${1:?usage: gate2.sh <label>}
BASE=$HOME/p2p-ab
OUT=$BASE/results
PORT=8899
ENGINE_PID=""

# An engine left holding both GPUs is a worse outcome than a failed check, and the first
# attempt did exactly that. Clean up on any exit path, not just the happy one.
cleanup() {
  if [ -n "$ENGINE_PID" ] && kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "cleaning up engine pid $ENGINE_PID"
    kill "$ENGINE_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$ENGINE_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$ENGINE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "$OUT"
cd "$HOME/vllm-env"
# shellcheck disable=SC1091
. .venv/bin/activate
set -a; . "$BASE/agent.env"; set +a
export VLLM_SERVER_DEV_MODE=1

run_one() {
  local name=$1 cfg=$2
  echo "=== $name: starting engine ==="
  vllm serve --config "$cfg" --port $PORT > "$BASE/gate2-$name.log" 2>&1 &
  ENGINE_PID=$!

  local ready=""
  for _ in $(seq 1 120); do
    if curl -sf --max-time 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
      echo "engine died during startup; tail of $BASE/gate2-$name.log:" >&2
      tail -20 "$BASE/gate2-$name.log" >&2
      return 1
    fi
    sleep 10
  done
  [ -n "$ready" ] || { echo "engine never became ready" >&2; return 1; }

  # Record what the engine actually resolved, so the label is not the only evidence that
  # the two runs differed in the way they were supposed to. Advisory: a failure to read it
  # must not abort the check it is only annotating.
  curl -s --max-time 5 "http://localhost:$PORT/server_info" 2>/dev/null \
    | python3 -c 'import json,sys,re
try:
    c = json.load(sys.stdin)["vllm_config"]
    m = re.search(r"disable_custom_all_reduce=([^,)]+)", c)
    print("  engine resolved disable_custom_all_reduce =", m.group(1) if m else "?")
except Exception as exc:
    print("  (could not read /server_info: %s)" % exc)' || true

  python3 "$BASE/greedy_capture.py" "$OUT/gate2-$LABEL-$name.json" "http://localhost:$PORT"

  cleanup
  ENGINE_PID=""
  sleep 8
}

run_one nccl      "$BASE/p2p-tp2-customar-off.yaml"
run_one custom-ar "$BASE/p2p-tp2-customar-on.yaml"

echo
echo "=== GATE 2 ==="
a=$(python3 -c "import json;print(json.load(open('$OUT/gate2-$LABEL-nccl.json'))['combined_sha256'])")
b=$(python3 -c "import json;print(json.load(open('$OUT/gate2-$LABEL-custom-ar.json'))['combined_sha256'])")
echo "  NCCL path      : $a"
echo "  custom all-red.: $b"
if [ "$a" = "$b" ]; then
  echo "  PASS — identical tokens"
else
  echo "  DIVERGED — inspect before trusting any throughput figure"
  python3 - "$OUT/gate2-$LABEL-nccl.json" "$OUT/gate2-$LABEL-custom-ar.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))["completions"]
b = json.load(open(sys.argv[2]))["completions"]
for x, y in zip(a, b):
    if x["sha256"] != y["sha256"]:
        print("  differs:", x["prompt"][:60])
PY
  exit 1
fi
