"""TransformerLens causal-tracing (activation patching) sweep.

The same study vLLM's ``/v1/patch_sweep`` runs in one call: patch the clean
run's ``resid_post`` (== vLLM ``post_block``) into the corrupt run at every
``(layer, position)`` cell and grade the answer token's logprob at the final
position.

Two variants:

* ``naive``   — one forward per cell (the loop a researcher actually writes).
* ``batched`` — one forward per layer, cells batched across positions and
  chunked so the full ``[batch, seq, vocab]`` logits fit in memory (unchunked,
  a 204-token prompt materializes ~12 GB of logits on a 151k vocab).
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any


def load_model(model_id: str, dtype: str = "bfloat16"):
    """Load a HookedTransformer on cuda (lazy transformer_lens import)."""
    import torch
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        model_id, dtype=getattr(torch, dtype), device="cuda"
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def run_patch_sweep(
    model,
    clean: str,
    corrupt: str,
    answer: str,
    variant: str,
    logits_chunk_budget: int = 4096,
) -> dict[str, Any]:
    """Run the full ``(layers x prompt positions)`` denoising sweep.

    Returns wall time, cells/s, baselines, and the recovered-metric argmax
    (used to cross-check agreement with the vLLM sweep endpoint).
    """
    import torch
    from transformer_lens import utils

    clean_toks = model.to_tokens(clean)
    corrupt_toks = model.to_tokens(corrupt)
    assert clean_toks.shape == corrupt_toks.shape, "equal-length pair required"
    n_pos = corrupt_toks.shape[1]
    n_layers = model.cfg.n_layers

    answer_ids = model.to_tokens(answer, prepend_bos=False)[0]
    assert answer_ids.numel() == 1, "answer must be a single token"
    answer_id = answer_ids.item()

    def answer_logprob(logits: torch.Tensor) -> float:
        return torch.log_softmax(logits[0, -1].float(), dim=-1)[answer_id].item()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    names = {utils.get_act_name("resid_post", layer) for layer in range(n_layers)}
    clean_logits, clean_cache = model.run_with_cache(
        clean_toks, names_filter=lambda n: n in names
    )
    clean_val = answer_logprob(clean_logits)
    corrupt_val = answer_logprob(model(corrupt_toks))

    grid = [[0.0] * n_pos for _ in range(n_layers)]

    if variant == "naive":

        def patch_one(resid, hook, pos, clean_resid):
            resid[:, pos] = clean_resid[:, pos]
            return resid

        for layer in range(n_layers):
            name = utils.get_act_name("resid_post", layer)
            for pos in range(n_pos):
                logits = model.run_with_hooks(
                    corrupt_toks,
                    fwd_hooks=[
                        (
                            name,
                            partial(
                                patch_one, pos=pos, clean_resid=clean_cache[name]
                            ),
                        )
                    ],
                )
                grid[layer][pos] = answer_logprob(logits)
    elif variant == "batched":
        chunk = min(n_pos, max(1, logits_chunk_budget // n_pos))

        def patch_rows(resid, hook, positions, clean_resid):
            for row, pos in enumerate(positions):
                resid[row, pos] = clean_resid[0, pos]
            return resid

        for layer in range(n_layers):
            name = utils.get_act_name("resid_post", layer)
            for c0 in range(0, n_pos, chunk):
                positions = list(range(c0, min(c0 + chunk, n_pos)))
                batch_toks = corrupt_toks.repeat(len(positions), 1)
                logits = model.run_with_hooks(
                    batch_toks,
                    fwd_hooks=[
                        (
                            name,
                            partial(
                                patch_rows,
                                positions=positions,
                                clean_resid=clean_cache[name],
                            ),
                        )
                    ],
                )
                lp = torch.log_softmax(logits[:, -1].float(), dim=-1)[:, answer_id]
                for row, pos in enumerate(positions):
                    grid[layer][pos] = lp[row].item()
    else:
        raise ValueError(f"unknown variant {variant!r}")

    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    denom = clean_val - corrupt_val
    rec = [[(v - corrupt_val) / denom for v in row] for row in grid]
    best = max((rec[i][j], i, j) for i in range(n_layers) for j in range(n_pos))
    return {
        "variant": f"tl_{variant}",
        "cells": n_layers * n_pos,
        "n_layers": n_layers,
        "n_positions": n_pos,
        "wall_s": round(wall, 3),
        "cells_per_s": round(n_layers * n_pos / wall, 1),
        "clean_logprob": round(clean_val, 4),
        "corrupt_logprob": round(corrupt_val, 4),
        "argmax": {
            "layer": best[1],
            "position": best[2],
            "recovered": round(best[0], 4),
        },
    }
