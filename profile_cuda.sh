  # Baseline: disabled
  nsys profile --stats=true --output=trace_disabled.nsys-rep \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    uv run --no-sync python scripts/bench_latency.py \
      --batch-sizes 16 --iters 3 --warmup 1 \
      --disable-prefix-cache --tag nsys-disabled

  # Steering, all 16 active
  nsys profile --stats=true --output=trace_all16.nsys-rep \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    uv run --no-sync python scripts/bench_latency.py \
      --batch-sizes 16 --iters 3 --warmup 1 \
      --tag nsys-all16

  nsys stats --report cuda_gpu_kern_sum trace_disabled.nsys-rep > kernels_disabled.txt
  nsys stats --report cuda_gpu_kern_sum trace_all16.nsys-rep > kernels_all16.txt
  diff kernels_disabled.txt kernels_all16.txt
