#!/usr/bin/env python3
"""Async streaming serving benchmark for capture-overhead measurement.

Measures TTFT, TPOT, E2EL, and throughput against an OpenAI-compatible
/v1/completions endpoint. Uniform code path across conditions; only the
optional ``capture`` field differs. Closed-loop fixed concurrency,
streaming (stream=true) for per-token timing, ignore_eos + fixed
max_tokens so output length is constant and TPOT comparable. Each prompt
gets a unique prefix so the prefix cache never short-circuits prefill.
"""
import argparse
import asyncio
import json
import statistics
import time

import aiohttp

BASE = (
    "In a distant land beyond the mountains there lived a community of "
    "scholars who spent their days studying the movements of the stars, "
    "the growth of plants, the flow of rivers, and the habits of the many "
    "creatures that shared their valley. Each morning they gathered to "
    "discuss what they had learned and to plan the experiments of the day. "
)


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


async def one(session, url, model, prompt, out_len, capture, rid, sem, results):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": out_len,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    if capture is not None:
        body["capture"] = {
            "filesystem": {
                "request_id": f"{capture['tag']}-{rid}",
                "tag": capture["tag"],
                "hooks": capture["hooks"],
                "positions": "last_prompt",
            }
        }
    async with sem:
        send = time.perf_counter()
        first = None
        last = send
        ntok = 0
        err = None
        try:
            async with session.post(url, json=body) as resp:
                if resp.status != 200:
                    err = f"HTTP {resp.status}: {(await resp.text())[:120]}"
                else:
                    async for raw in resp.content:
                        line = raw.decode("utf-8", "ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except Exception:
                            continue
                        txt = obj.get("choices", [{}])[0].get("text", "")
                        if txt:
                            now = time.perf_counter()
                            if first is None:
                                first = now
                            last = now
                            ntok += 1
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        end = time.perf_counter()
        results.append(
            dict(send=send, first=first, last=last, end=end, ntok=ntok, err=err)
        )


async def run(args):
    capture = None
    if args.capture:
        capture = {"tag": args.tag, "hooks": json.loads(args.hooks)}
    prompts = [f"Request {i} unique-{i * 7919 % 100003}. " + BASE for i in range(args.n)]
    sem = asyncio.Semaphore(args.concurrency)
    results = []
    url = args.url.rstrip("/") + "/v1/completions"
    timeout = aiohttp.ClientTimeout(total=args.req_timeout)
    conn = aiohttp.TCPConnector(limit=args.concurrency + 8)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as s:
        # warmup (not measured)
        await asyncio.gather(*[
            one(s, url, args.model, prompts[i], min(16, args.out_len), capture,
                f"warm{i}", sem, [])
            for i in range(min(args.warmup, args.n))
        ])
        t0 = time.perf_counter()
        await asyncio.gather(*[
            one(s, url, args.model, prompts[i], args.out_len, capture, i, sem, results)
            for i in range(args.n)
        ])
        t1 = time.perf_counter()
    ok = [r for r in results if r["err"] is None and r["first"] is not None]
    errs = [r["err"] for r in results if r["err"] is not None]
    ttft = [(r["first"] - r["send"]) * 1000 for r in ok]
    e2el = [(r["last"] - r["send"]) * 1000 for r in ok]
    tpot = [
        (r["last"] - r["first"]) * 1000 / (r["ntok"] - 1)
        for r in ok if r["ntok"] > 1
    ]
    tot_out = sum(r["ntok"] for r in ok)
    wall = t1 - t0
    summary = dict(
        label=args.label, n=args.n, ok=len(ok), errors=len(errs),
        concurrency=args.concurrency, out_len=args.out_len,
        ttft_ms=dict(mean=statistics.mean(ttft) if ttft else None,
                     p50=pct(ttft, 50), p99=pct(ttft, 99)),
        tpot_ms=dict(mean=statistics.mean(tpot) if tpot else None,
                     p50=pct(tpot, 50), p99=pct(tpot, 99)),
        e2el_ms=dict(mean=statistics.mean(e2el) if e2el else None,
                     p50=pct(e2el, 50), p99=pct(e2el, 99)),
        out_tok_per_s=tot_out / wall if wall else None,
        req_per_s=len(ok) / wall if wall else None,
        wall_s=wall,
        sample_errors=errs[:3],
    )
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(summary) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://node0:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--req-timeout", type=float, default=600)
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--hooks", default='{"post_mlp": "all"}')
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
