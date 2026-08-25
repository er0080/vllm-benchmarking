#!/bin/bash
# Test the upstream workaround as an intervention, not an assertion.
#
# pytorch/pytorch#158029: NCCL auto-registers collective buffers under CUDA graph capture and
# assumes one contiguous allocation. Expandable segments back a reserved VA range with lazily
# mapped chunks, so that assumption is false. NCCL_GRAPH_REGISTER=0 disables the registration.
#
# Prediction: every arm below passes with expandable_segments still ON. If the prediction fails
# the mechanism is wrong, whatever the upstream issue says about it.
set -uo pipefail
BASE=$HOME/p2p-ab
LOG=$BASE/reg-cross.log

# Queue behind the running cross arms rather than fighting them for the GPUs.
for _ in $(seq 1 120); do
  grep -q "^=== DONE ===" "$BASE/exp-cross.log" 2>/dev/null && break
  sleep 30
done
: > "$LOG"

run() {
  local label=$1 cfg=$2 nccl=$3
  echo "--- $label: NCCL_GRAPH_REGISTER=0, expandable_segments ON ---" >> "$LOG"
  NCCL_GRAPH_REGISTER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash -ic "$BASE/arm2.sh $label $BASE/$cfg $nccl" >> "$LOG" 2>&1
  sleep 10
}

# The arm that failed 0/8 an hour ago, and the 0.95 production-shaped one that hung five times.
run reg0-090-sys mtp090-base.yaml SYS
run reg0-095-sys mtp095-base.yaml SYS
run reg0-090-both mtp090-push.yaml SYS

echo "=== DONE ===" >> "$LOG"
