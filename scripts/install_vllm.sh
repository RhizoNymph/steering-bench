#!/bin/bash
# Install vLLM editable against a local worktree, using the precompiled C++
# wheel pinned to a specific commit. Avoids a ~30 min source build and the
# ABI-mismatch errors you get when `VLLM_USE_PRECOMPILED` is set without
# also pinning the wheel commit.
#
# Run from the steering-bench checkout. Expects the vllm worktree to be a
# sibling at ../vllm/<branch> (default ../vllm/steering).
#
# Overrides via env:
#   VLLM_WORKTREE   — path to the vllm checkout (default ../vllm/steering)
#   WHEEL_COMMIT    — wheel SHA to pin against (default known-good 21943d4c)
#   TORCH_BACKEND   — uv torch-backend (default cu129; use cu130 on CUDA 13)
#
# Pick TORCH_BACKEND from `nvidia-smi | grep "CUDA Version"`:
#   CUDA 12.x  → cu129
#   CUDA 13.x  → cu130
# (Per vllm/steering/setup.py: supported = {12: "cu129", 13: "cu130"})
#
# The wheel-commit SHA is *the* footgun: if it drifts from the local
# Python source it imports blow up with "undefined symbol" or
# "module has no attribute". When bumping vllm, update the pin to
# whatever the upstream wheel publisher currently has.

set -euo pipefail

VLLM_WORKTREE="${VLLM_WORKTREE:-../vllm/steering}"
WHEEL_COMMIT="${WHEEL_COMMIT:-21943d4c258983c4b8eb56d50029aca4f18e4629}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"

if [ ! -d "$VLLM_WORKTREE" ]; then
  echo "vllm worktree not found at: $VLLM_WORKTREE" >&2
  echo "Set VLLM_WORKTREE=/path/to/vllm/checkout" >&2
  exit 1
fi

echo "Installing vllm:"
echo "  worktree:     $VLLM_WORKTREE"
echo "  wheel commit: $WHEEL_COMMIT"
echo "  torch backend: $TORCH_BACKEND"
echo ""

VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT="$WHEEL_COMMIT" \
uv pip install -e "$VLLM_WORKTREE" --torch-backend="$TORCH_BACKEND"
