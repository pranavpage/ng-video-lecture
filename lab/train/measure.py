"""
Measure a trained checkpoint: what it does with a held-out prompt, and how fast.

For each model we feed the last block_size characters of a fixed validation-split prompt
(ending "Therefore, my lor"), then:
  - ingestion: time one forward pass over the whole prompt (prefill), median of N repeats
  - probe:     the next-character distribution and every head's attention for the last position
  - generation: continue the prompt one token at a time like gpt.py's generate() (batch 1,
               full-context forward per token, no KV cache), recording each step's sampled
               character, top-k candidates and its own wall-clock latency

usage:
    python measure.py                    # measure every checkpoint that has no measurement yet
    python measure.py --force L2_H2_E64_T32
"""
import argparse
import json
import os
import statistics
import time

import torch
from torch.nn import functional as F

from sweep import (CKPT_DIR, RUNS_DIR, PROBE_TEXT, PROBE_TEXT_NEXT, GPTLanguageModel,
                   decode, device, encode, sync)

SETTINGS = dict(
    gen_tokens=300,
    topk=8,
    prefill_repeats=30,
    warmup=5,
    seed=1337,
    batch=1,
    method='batch 1, no KV cache: one full-context forward pass per generated token, as in gpt.py; '
            'sampling on CPU with a fixed seed; latency per token includes the sampling step',
)


def measure_path(rid):
    return os.path.join(RUNS_DIR, rid + '.measure.json')


def load(rid):
    ck = torch.load(os.path.join(CKPT_DIR, rid + '.pt'), map_location=device)
    c = ck['config']
    model = GPTLanguageModel(c['n_layer'], c['n_head'], c['n_embd'], c['block_size'], 0.0).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, c


@torch.no_grad()
def measure(rid):
    S = SETTINGS
    model, c = load(rid)
    T = c['block_size']
    prompt = PROBE_TEXT[-T:]
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    # ---- ingestion: prefill the whole prompt in one forward pass ----
    for _ in range(S['warmup']):
        model(idx)
    sync()
    times = []
    for _ in range(S['prefill_repeats']):
        t0 = time.perf_counter()
        model(idx)
        sync()
        times.append(time.perf_counter() - t0)
    prefill_s = statistics.median(times)

    # ---- probe: what the model predicts after the prompt, and where its heads look ----
    logits, _ = model(idx)
    probs = F.softmax(logits[0, -1], dim=-1).cpu().tolist()
    attn = [[[round(w, 4) for w in h.last_wei[0, -1].cpu().tolist()] for h in blk.sa.heads]
            for blk in model.blocks]

    # ---- generation: continue the prompt token by token, timing each step ----
    g = torch.Generator().manual_seed(S['seed'])
    ctx = idx
    for _ in range(S['warmup']):
        model(ctx[:, -T:])
    sync()
    steps = []
    for _ in range(S['gen_tokens']):
        t0 = time.perf_counter()
        logits, _ = model(ctx[:, -T:])
        p = F.softmax(logits[0, -1], dim=-1).cpu()
        nxt = torch.multinomial(p, num_samples=1, generator=g)
        dt = time.perf_counter() - t0
        top = torch.topk(p, S['topk'])
        # [sampled index, ms, p(sampled), [[index, p], ...top-k]]
        steps.append([int(nxt), round(dt * 1000, 3), round(float(p[nxt]), 4),
                      [[int(j), round(float(q), 4)] for q, j in zip(top.values, top.indices)]])
        ctx = torch.cat([ctx, nxt.view(1, 1).to(device)], dim=1)
    gen_ms = [s[1] for s in steps]

    out = dict(
        speed=dict(
            prefill_ms=round(prefill_s * 1000, 3),
            prefill_tok_s=round(T / prefill_s, 1),
            gen_ms_per_tok=round(statistics.median(gen_ms), 3),
            gen_tok_s=round(1000 / statistics.median(gen_ms), 1),
            gen_total_s=round(sum(gen_ms) / 1000, 3),
        ),
        detail=dict(
            id=rid,
            prompt=prompt,
            target=PROBE_TEXT_NEXT,
            probs=[round(q, 4) for q in probs],
            attn_last=attn,
            text=decode([s[0] for s in steps]),
            steps=steps,
        ),
    )
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(measure_path(rid), 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', nargs='*', default=None)
    args = ap.parse_args()
    for fn in sorted(os.listdir(CKPT_DIR)):
        rid = fn[:-3]
        if args.force is not None and args.force and rid not in args.force:
            continue
        if args.force is None and os.path.exists(measure_path(rid)):
            continue
        m = measure(rid)
        print(rid, m['speed'], repr(m['detail']['text'][:40]), flush=True)


if __name__ == '__main__':
    main()
