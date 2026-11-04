"""
Logits-equivalence checks for the KV cache in gpt.py. No timing.

Compares, with identical weights, eval() mode and no_grad:
  - "orig":    Karpathy's model from the pre-cache revision (git d3a418c~1)
  - "full":    my model, full forward with past_kv=None
  - "cached":  my model, prefill + cached decode

gpt.py trains at import time, so instead of importing it this script execs only
the part of each file above the model/optimizer setup (hyperparameters, data,
model classes) into its own namespace.

usage: python check_kv_cache.py [--device cpu|mps] [--seed 0]
"""
import argparse
import subprocess
import torch

ORIG_REV = "d3a418c~1"
ATOL = 1e-5


def load_defs(source, cut_marker):
    """exec everything in `source` before `cut_marker`; return the namespace."""
    prefix = source[: source.index(cut_marker)]
    ns = {"__name__": "gpt_defs"}
    exec(compile(prefix, "gpt_defs", "exec"), ns)
    return ns


def max_diff(a, b):
    return (a - b).abs().max().item()


def report(name, a, b):
    d = max_diff(a, b)
    ok = torch.allclose(a, b, atol=ATOL, rtol=0)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<58} max|diff| = {d:.2e}")
    return ok


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = args.device

    new_src = open("gpt.py").read()
    orig_src = subprocess.run(["git", "show", f"{ORIG_REV}:gpt.py"],
                              capture_output=True, text=True, check=True).stdout

    new = load_defs(new_src, "# Model and optimizer setup")
    orig = load_defs(orig_src, "model = GPTLanguageModel()")
    # both files read `device` as a global at call time; point them at ours
    new["device"] = dev
    orig["device"] = dev

    torch.manual_seed(args.seed)
    m = new["GPTLanguageModel"]().to(dev).eval()
    m0 = orig["GPTLanguageModel"]().to(dev).eval()
    m0.load_state_dict(m.state_dict())  # same parameter names in both revisions

    V, BS = new["vocab_size"], new["block_size"]
    B = 2
    print(f"device={dev} seed={args.seed} B={B} block_size={BS} vocab={V} "
          f"weights=random(untrained) dtype={next(m.parameters()).dtype} atol={ATOL}")
    results = []

    # 1. full forward: my model (past_kv=None) vs Karpathy's original
    print("\n1. full forward, my model vs original")
    for N in [1, 2, 20, BS]:
        idx = torch.randint(V, (B, N), device=dev)
        a, _, _ = m(idx)
        b, _ = m0(idx)
        results.append(report(f"N={N}, all positions", a, b))

    # 2. prefill N-1, then one cached token; compare last-position logits
    print("\n2. prefill N-1 + one cached token vs original full forward (last position)")
    for N in [2, 20, BS]:
        idx = torch.randint(V, (B, N), device=dev)
        _, _, cache = m(idx[:, :-1])
        step, _, _ = m(idx[:, -1:], past_kv=cache)
        ref, _ = m0(idx)
        results.append(report(f"N={N}", step[:, -1], ref[:, -1]))

    # 3. token-by-token decode over a whole block; compare every step
    print("\n3. prefill P, then decode one token at a time to block_size")
    for P in [1, 10]:
        idx = torch.randint(V, (B, BS), device=dev)
        ref, _ = m0(idx)
        logits, _, cache = m(idx[:, :P])
        rows = [logits]
        for t in range(P, BS):
            logits, _, cache = m(idx[:, t:t + 1], past_kv=cache)
            rows.append(logits)
        results.append(report(f"P={P}, all {BS} positions", torch.cat(rows, dim=1), ref))

    # 4. chunked append: T>1 new tokens onto a non-empty cache (the offset-mask path)
    print("\n4. chunked append: prefill P, then a chunk of T tokens at once")
    for P, T in [(1, 2), (10, 5), (100, 50), (200, 56)]:
        idx = torch.randint(V, (B, P + T), device=dev)
        ref, _ = m0(idx)
        _, _, cache = m(idx[:, :P])
        chunk, _, _ = m(idx[:, P:], past_kv=cache)
        results.append(report(f"P={P}, T={T}, positions P..P+T-1", chunk, ref[:, P:]))

    # 5. greedy generation: cached vs original full-recompute loop
    print("\n5. greedy generation, cached vs original full recompute")
    prompt_len, n_new = 10, 200
    prompt = torch.randint(V, (B, prompt_len), device=dev)

    seq0 = prompt
    for _ in range(n_new):
        logits, _ = m0(seq0[:, -BS:])
        seq0 = torch.cat([seq0, logits[:, -1].argmax(-1, keepdim=True)], dim=1)

    seq1 = prompt
    logits, _, cache = m(prompt)
    for step in range(n_new):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        seq1 = torch.cat([seq1, nxt], dim=1)
        if step == n_new - 1:
            break
        logits, _, cache = m(nxt, past_kv=cache)

    same = (seq0[:, prompt_len:] == seq1[:, prompt_len:])
    ok = bool(same.all())
    results.append(ok)
    first = "none" if ok else str((~same).nonzero()[0].tolist())
    print(f"  {'PASS' if ok else 'FAIL'}  prompt={prompt_len}, new={n_new}: "
          f"{same.sum().item()}/{same.numel()} tokens identical, first mismatch [b, t]: {first}")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
