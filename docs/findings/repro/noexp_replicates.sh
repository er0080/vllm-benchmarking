#!/bin/bash
# Five replicates of the arm the new hypothesis says should be clean:
# MTP + NCCL_P2P_LEVEL=SYS + CUDA graphs + expandable_segments OFF.
#
# Two arms passed under these conditions, which is not enough to call it clean. The
# non-speculative sweeps show 2 engine crashes in 66 SYS runs with expandable segments absent,
# so a residual P2P fragility independent of the allocator is still live. MTP amplified the
# expandable-segments failure from occasional to certain; if a residual exists it should
# amplify that too, and five replicates will see it if it is anywhere near that rate.
set -uo pipefail
BASE=$HOME/p2p-ab
LOG=$BASE/rep-noexp.log

for _ in $(seq 1 240); do
  grep -q "^=== DONE ===" "$BASE/reg-cross.log" 2>/dev/null && break
  sleep 30
done
: > "$LOG"

for i in 1 2 3 4 5; do
  # PYTORCH_CUDA_ALLOC_CONF is commented out of ~/.bashrc, so bash -ic leaves it unset.
  bash -ic "$BASE/arm2.sh noexp-sys-r$i $BASE/mtp090-base.yaml SYS" >> "$LOG" 2>&1
  sleep 10
done

echo "=== DONE ===" >> "$LOG"
