#!/usr/bin/env bash
# Run a focused nsys trace around a single steering bench mode.
#
# Strategy:
#   1. Launch the vLLM server under ``nsys profile`` with
#      ``--capture-range=cudaProfilerApi``.  nsys idles in the
#      background (no recording) until the server calls
#      ``cudaProfilerStart``.
#   2. Wait for the server to be ready (model load + cudagraph capture
#      both happen during the no-recording phase so the trace stays
#      small).
#   3. POST to ``/v1/steering/_profile/cuda_start`` to begin recording.
#   4. Fire a tiny burst of concurrent requests against
#      ``/v1/completions`` — that's the slice we want captured.
#   5. POST to ``/v1/steering/_profile/cuda_stop`` to end recording.
#   6. Tear the server down.  nsys flushes the ``.nsys-rep`` to disk.
#
# Open the resulting file with ``nsys-ui`` (or upload elsewhere).
# Steering hot paths show up under NVTX ranges named
# ``manager.populate.*``, ``mixin.*``, ``kernel.apply_steering_triton.*``.
#
# Requires ``nsys`` on PATH (NVIDIA Nsight Systems CLI).
#
# Env vars:
#   MODEL=...            HF model id (default: google/gemma-3-4b-it)
#   MODE=...             steering mode (default: per_request_n4)
#                        one of: enabled_idle, named_shared,
#                        all_steered_shared, per_request_n4
#   NUM_PROMPTS=...      requests to send during the recording window
#   CONCURRENCY=...      concurrent requests
#   OUT=...              output base name (default: results/3090-timing/nsys/<mode>)
#   PYTHON=...           interpreter (default: .venv/bin/python)
#   PORT=...             server port (default: 8765)
#
# Usage:
#   scripts/nsys_steering_profile.sh
#   MODE=named_shared NUM_PROMPTS=8 scripts/nsys_steering_profile.sh

set -euo pipefail

if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: nsys not on PATH. Install Nsight Systems CLI." >&2
    exit 1
fi

MODEL="${MODEL:-google/gemma-3-4b-it}"
MODE="${MODE:-per_request_n4}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
CONCURRENCY="${CONCURRENCY:-4}"
PORT="${PORT:-8765}"
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_BASE="${OUT:-results/3090-timing/nsys/${MODE}}"

# Defaults for the bench shape — keep small so the trace is small.
MAX_TOKENS="${MAX_TOKENS:-64}"
PROMPT_LEN="${PROMPT_LEN:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_STEERING_CONFIGS="${MAX_STEERING_CONFIGS:-16}"

SERVER_LOG="/tmp/vllm_serving_nsys.log"
NSYS_REP="${OUT_BASE}.nsys-rep"

mkdir -p "$(dirname "$OUT_BASE")"

echo "[nsys-profile] model=$MODEL mode=$MODE n=$NUM_PROMPTS c=$CONCURRENCY"
echo "[nsys-profile] server log: $SERVER_LOG"
echo "[nsys-profile] trace:      $NSYS_REP"

SERVER_ENV=(
    env
    VLLM_STEERING_TIMING=1
    VLLM_STEERING_NVTX=1
    VLLM_SERVER_DEV_MODE=1
)

# Capture range driven by torch.cuda.profiler.{start,stop}.  ``-t``
# selects the trace categories; cuda+nvtx is enough.
NSYS_CMD=(
    nsys profile
    -o "$OUT_BASE"
    -t cuda,nvtx
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --cuda-memory-usage=true
    --force-overwrite=true
    --stop-on-exit=true
)

# --- launch server under nsys ----------------------------------------------
echo "[nsys-profile] launching server under nsys..."
"${SERVER_ENV[@]}" "${NSYS_CMD[@]}" \
    "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --enable-steering \
    --max-steering-configs "$MAX_STEERING_CONFIGS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[nsys-profile] server pid=$SERVER_PID"

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[nsys-profile] killing server pid=$SERVER_PID"
        kill -INT "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- wait for ready --------------------------------------------------------
echo "[nsys-profile] waiting for server (model load + torch.compile + cudagraph)..."
for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[nsys-profile] server ready after ${i}s"
        break
    fi
    sleep 1
    if [[ $i == 240 ]]; then
        echo "[nsys-profile] timed out waiting for server"
        tail -50 "$SERVER_LOG" >&2
        exit 1
    fi
done

# --- prepare per-request bodies (mode-dependent) --------------------------
# Driven entirely by a Python helper so we get the same vector spec the
# bench would use.  Writes a JSON list of {prompt, extra_body} pairs to
# a temp file that the next step posts in parallel.
REQUESTS_JSON="$(mktemp --suffix=.json)"
"$PYTHON" - <<PY >"$REQUESTS_JSON"
import json, sys
sys.path.insert(0, "src")
from steering_bench.vectors import (
    random_steering_vectors, random_steering_vectors_diverse,
)

mode = "$MODE"
n = $NUM_PROMPTS
prompt_len = $PROMPT_LEN

# Synthetic prompts.
words = max(1, int(prompt_len / 1.3))
prompt = " ".join(["hello"] * words)

# Gemma-3-4b-it constants — adequate for the on-3090 profiling use case.
HIDDEN_SIZE = 2560
NUM_LAYERS = 34

shared = random_steering_vectors(
    hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
    hook_points=["post_mlp"], scale=0.1, seed=42,
)
def _coerce(v):
    # ``random_steering_vectors`` already returns plain Python lists,
    # but normalize the layer index key to int to satisfy the
    # /v1/steering/modules/register schema.
    out = {}
    for hp, layer_vecs in v.items():
        out[hp] = {}
        for k, val in layer_vecs.items():
            if hasattr(val, "tolist"):
                val = val.tolist()
            out[hp][int(k)] = val
    return out
shared_jsonable = _coerce(shared)

def build_body(i):
    if mode == "enabled_idle":
        return None
    if mode == "all_steered_shared":
        return {"steering_vectors": shared_jsonable}
    if mode == "named_shared":
        return {"steering_name": "bench_named_shared"}
    if mode.startswith("per_request_n"):
        k = int(mode.split("_n")[-1])
        diverse = random_steering_vectors_diverse(
            hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
            num_configs=k, hook_points=["post_mlp"],
            scale=0.1, base_seed=42,
        )
        return {"steering_vectors": _coerce(diverse[i % k])}
    raise ValueError(f"unknown mode: {mode!r}")

req_specs = [
    {"prompt": prompt, "extra_body": build_body(i)} for i in range(n)
]
json.dump(req_specs, sys.stdout)
PY

# Register the named module if needed.
if [[ "$MODE" == "named_shared" ]]; then
    echo "[nsys-profile] registering named module bench_named_shared"
    "$PYTHON" - <<PY
import json, sys
sys.path.insert(0, "src")
import httpx
from steering_bench.vectors import random_steering_vectors
v = random_steering_vectors(
    hidden_size=2560, num_layers=34, hook_points=["post_mlp"],
    scale=0.1, seed=42,
)
def _coerce_layer(val):
    return val.tolist() if hasattr(val, "tolist") else val
vectors = {hp: {int(k): _coerce_layer(val) for k, val in layer_vecs.items()}
           for hp, layer_vecs in v.items()}
payload = {"name": "bench_named_shared", "vectors": vectors,
           "prefill_vectors": None, "decode_vectors": None}
r = httpx.post("http://127.0.0.1:${PORT}/v1/steering/modules/register",
               json=payload, timeout=60.0)
r.raise_for_status()
print("registered:", r.status_code)
PY
fi

# --- start the nsys capture window ----------------------------------------
echo "[nsys-profile] starting cuda profiler (capture begins now)"
curl -sf -X POST "http://127.0.0.1:${PORT}/v1/steering/_profile/cuda_start" >/dev/null

# --- fire the requests ----------------------------------------------------
echo "[nsys-profile] sending $NUM_PROMPTS requests (concurrency=$CONCURRENCY)..."
"$PYTHON" - <<PY
import asyncio, json, sys
import httpx

with open("$REQUESTS_JSON") as f:
    specs = json.load(f)

CONCURRENCY = $CONCURRENCY
MAX_TOKENS = $MAX_TOKENS
MODEL = "$MODEL"
URL = "http://127.0.0.1:${PORT}/v1/completions"

sem = asyncio.Semaphore(CONCURRENCY)

async def one(client, i, spec):
    async with sem:
        body = {
            "model": MODEL,
            "prompt": spec["prompt"],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "stream": True,
        }
        if spec["extra_body"]:
            body.update(spec["extra_body"])
        async with client.stream("POST", URL, json=body) as r:
            r.raise_for_status()
            async for _ in r.aiter_lines():
                pass

async def main():
    async with httpx.AsyncClient(timeout=120.0) as client:
        await asyncio.gather(*[one(client, i, s) for i, s in enumerate(specs)])

asyncio.run(main())
print("all requests done")
PY

# --- stop the nsys capture window -----------------------------------------
echo "[nsys-profile] stopping cuda profiler"
curl -sf -X POST "http://127.0.0.1:${PORT}/v1/steering/_profile/cuda_stop" >/dev/null

rm -f "$REQUESTS_JSON"

# --- tear down -------------------------------------------------------------
cleanup
trap - EXIT

# Brief wait for nsys to flush.
sleep 3
echo "[nsys-profile] done"
echo "[nsys-profile] trace: $NSYS_REP"
echo "[nsys-profile] open with: nsys-ui $NSYS_REP"
