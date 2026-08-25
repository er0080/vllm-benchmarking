#!/bin/bash
# Cross expandable_segments against the P2P transport, holding gpu-memory-utilization at 0.90.
#
# The attribution arms showed PYTORCH_CUDA_ALLOC_CONF, not gpu-memory-utilization, decides
# whether NCCL_P2P_LEVEL=SYS survives an MTP run. These arms ask what else it takes: whether
# CUDA graphs are required for the failure, and whether the push kernel is exposed to it too.
# Every arm has expandable_segments ON, which is the failing side of that pair.
set -uo pipefail
BASE=$HOME/p2p-ab
LOG=$BASE/exp-cross.log
: > "$LOG"

sed 's/^disable-custom-all-reduce: true$/disable-custom-all-reduce: true\nenforce-eager: true/' \
  "$BASE/mtp090-base.yaml" > "$BASE/mtp090-eager.yaml"

run() {
  local label=$1 cfg=$2 nccl=$3
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash -ic "$BASE/arm2.sh $label $BASE/$cfg $nccl" >> "$LOG" 2>&1
  sleep 10
}

run exp-off       mtp090-base.yaml  off
run exp-sys-eager mtp090-eager.yaml SYS
run exp-push      mtp090-push.yaml  off
run exp-both      mtp090-push.yaml  SYS

echo "=== DONE ===" >> "$LOG"
