#!/usr/bin/env bash
# h100-setup.sh — bring a rental H100 box up to the steering-bench run state.
#
# Idempotent at the step level (re-running skips work already done).
# Reads HF_TOKEN from the environment for non-interactive HuggingFace login.
#
# Usage on the rental box:
#
#   # If you don't have steering-bench cloned yet, fetch this script alone:
#   curl -fsSL \
#     https://raw.githubusercontent.com/RhizoNymph/steering-bench/feat/steering-modes-matrix/scripts/h100-setup.sh \
#     -o h100-setup.sh
#   chmod +x h100-setup.sh
#   HF_TOKEN=hf_xxx ./h100-setup.sh
#
#   # Or if you already cloned the repo, just run it from inside:
#   cd /home/nymph/Code/steering-bench
#   HF_TOKEN=hf_xxx ./scripts/h100-setup.sh
#
# When done, run the bench:
#
#   cd /home/nymph/Code/steering-bench
#   SHAREGPT_PATH=~/data/ShareGPT_V3_unfiltered_cleaned_split.json \
#     ./scripts/run_h100.sh 2>&1 | tee run_h100.log

set -euo pipefail

# ─── 1. uv (system Python is not used) ──────────────────────────────────
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# ─── 2. layout — match the hardcoded path in steering-bench/pyproject.toml ─
sudo mkdir -p /home/nymph/Code
sudo chown -R "$USER:$USER" /home/nymph
cd /home/nymph/Code

# ─── 3. clone vllm fork at feat/steering (includes auto-promote PR #145) ─
if [[ ! -d /home/nymph/Code/vllm/.git ]]; then
    git clone --branch feat/steering \
        https://github.com/RhizoNymph/vllm.git
fi

# ─── 4. clone steering-bench, then switch to the modes-matrix branch ────
if [[ ! -d /home/nymph/Code/steering-bench/.git ]]; then
    git clone https://github.com/RhizoNymph/steering-bench.git
fi
cd /home/nymph/Code/steering-bench

# Checkout the branch that adds the named_shared serving mode and the
# steering modes matrix runner.  Rebase off remote so we pick up any
# fixes pushed since this script was last updated.
git fetch origin
if git rev-parse --verify --quiet feat/steering-modes-matrix >/dev/null; then
    git checkout feat/steering-modes-matrix
    git pull --ff-only origin feat/steering-modes-matrix
else
    git checkout -b feat/steering-modes-matrix \
        origin/feat/steering-modes-matrix
fi

# ─── 5. venv + editable installs ────────────────────────────────────────
if [[ ! -d .venv ]]; then
    uv venv --python 3.12
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# vllm fork first (Python-only changes — skip the C++ rebuild).
VLLM_USE_PRECOMPILED=1 uv pip install -e /home/nymph/Code/vllm \
    --torch-backend=auto

# steering-bench (depends on vllm).
uv pip install -e .

# ─── 6. HF auth (Gemma-3 is gated) ──────────────────────────────────────
uv pip install huggingface_hub
if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    huggingface-cli login
fi

# ─── 7. pre-download models so the bench doesn't block on downloads ─────
huggingface-cli download google/gemma-3-4b-it
huggingface-cli download google/gemma-3-27b-it

# ─── 8. ShareGPT (optional, ~600 MB) ────────────────────────────────────
mkdir -p ~/data
SHAREGPT_FILE=~/data/ShareGPT_V3_unfiltered_cleaned_split.json
if [[ ! -s "$SHAREGPT_FILE" ]]; then
    curl -L -o "$SHAREGPT_FILE" \
        https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
fi

# ─── 9. clock pinning (rental boxes usually deny this — best-effort) ───
# The run_h100.sh defaults to warmup=10 / iters=30 so variance averages
# out without locked clocks.  If sudo isn't available this is a no-op.
if sudo -n true 2>/dev/null; then
    sudo nvidia-smi -pm 1 || true
    # H100 SXM5 base/boost ≈ 1395/1980 MHz; PCIe ≈ 1095/1980.  Lock to
    # base for repeatability — if the box doesn't allow it the bench
    # script will note this in its output.
    sudo nvidia-smi -lgc 1395,1395 || true
fi

# ─── 10. smoke test (fail fast before the multi-hour bench) ─────────────
.venv/bin/python scripts/verify_correctness.py --model google/gemma-3-4b-it

cat <<'EOF'

setup complete.  Next:

  cd /home/nymph/Code/steering-bench
  SHAREGPT_PATH=~/data/ShareGPT_V3_unfiltered_cleaned_split.json \
    ./scripts/run_h100.sh 2>&1 | tee run_h100.log

The full sweep is ~8-9 hr on a single H100 80GB.  Override flags:

  DO_27B=0   skip the 27B section
  DO_MICRO=0 skip the kernel-level microbenchmarks
  DRY_RUN=1  preview the step list without running anything

EOF
