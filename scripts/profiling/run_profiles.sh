#!/bin/bash
# Orchestrate the capture-consumer profilers. Run a stage by arg:
#   run_profiles.sh {torch|nsys|pyspy|all}
#
# Env overrides (all optional):
#   PYTHON  python from the vllm-direct venv   (default: python)
#   NSYS    nsys binary                         (default: nsys)
#   PYSPY   py-spy binary                       (default: py-spy)
#   MODEL   model id                            (default: google/gemma-3-4b-it)
#   BS      batch size                          (default: 16)
set -u
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
NSYS=${NSYS:-nsys}
PYSPY=${PYSPY:-py-spy}
MODEL=${MODEL:-google/gemma-3-4b-it}
BS=${BS:-16}
HARNESS=profile_capture.py

stage_torch() {
  echo "### torch.profiler: baseline vs logging_max (op-level diff) ###"
  for cfg in baseline logging_max; do
    $PYTHON $HARNESS --config $cfg --mode torch --trace-dir "trace_$cfg" \
      --model "$MODEL" --batch-size $BS | grep -E "RESULT|trace"
    sleep 4
  done
  $PYTHON analyze_trace.py baseline=trace_baseline capture=trace_logging_max
}

stage_nsys() {
  # nsys traces the driver's NVTX; the worker's CUDA lives in the EngineCore
  # subprocess and is covered by --mode torch instead. --trace=cuda,nvtx only
  # (python-sampling / osrt need CPU sampling that may be unavailable).
  echo "### nsys: NVTX wall (gen_iter) for baseline vs logging_max ###"
  for cfg in baseline logging_max; do
    $NSYS profile -o "nsys_$cfg" --force-overwrite=true --trace=cuda,nvtx \
      $PYTHON $HARNESS --config $cfg --mode plain --iters 2 --warmup 2 \
      --model "$MODEL" --batch-size $BS | grep -E "RESULT"
    sleep 4
  done
  for cfg in baseline logging_max; do
    echo "== $cfg NVTX =="
    $NSYS stats --report nvtx_sum --format table "nsys_$cfg.nsys-rep" 2>/dev/null \
      | grep -E "gen_iter|timed_region"
  done
}

stage_pyspy() {
  echo "### py-spy: CPU flamegraph + folded stacks for logging_max ###"
  $PYSPY record --subprocesses --format flamegraph --rate 250 -o pyspy_logging_max.svg -- \
    $PYTHON $HARNESS --config logging_max --mode plain --iters 40 --warmup 3 \
    --model "$MODEL" --batch-size $BS
  $PYSPY record --subprocesses --format raw --rate 250 -o pyspy_logging_max.folded -- \
    $PYTHON $HARNESS --config logging_max --mode plain --iters 40 --warmup 3 \
    --model "$MODEL" --batch-size $BS
  $PYTHON agg_folded.py pyspy_logging_max.folded
}

case "${1:-all}" in
  torch) stage_torch ;;
  nsys)  stage_nsys ;;
  pyspy) stage_pyspy ;;
  all)   stage_torch; stage_nsys; stage_pyspy ;;
  *) echo "usage: $0 {torch|nsys|pyspy|all}"; exit 1 ;;
esac
