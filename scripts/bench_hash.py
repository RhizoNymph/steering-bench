import time
from vllm.config.steering_types import hash_steering_config
from steering_bench.vectors import random_steering_vectors
vectors = random_steering_vectors(
    hidden_size=2560, num_layers=34,
    hook_points=["post_mlp"], scale=0.1, seed=42,
)

# Warm up
for _ in range(3): hash_steering_config(vectors)
# Time it
t0 = time.perf_counter()
for _ in range(100):
    hash_steering_config(vectors)
t1 = time.perf_counter()
print(f"Current: {(t1-t0)/100*1000:.2f} ms/call")
