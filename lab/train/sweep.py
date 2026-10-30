"""
Precompute real training runs for TinyGPT Lab.

The model is the exact architecture from ../../gpt.py (per-head attention modules,
4x ReLU feedforward, pre-LayerNorm residual blocks, learned positional embeddings),
with n_layer / n_head / n_embd / block_size exposed so we can sweep a grid of them.
Every number the lab displays comes from JSON written by this script and measure.py
(which benchmarks each saved checkpoint right after it is trained).

usage:
    python sweep.py                         # run the whole grid (resumable)
    python sweep.py --only L2_H2_E64_T32    # run a single config
"""
import argparse
import itertools
import json
import os
import platform
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, 'runs')
CKPT_DIR = os.path.join(HERE, 'ckpt')
OUT_JSON = os.path.join(HERE, '..', 'web', 'public', 'runs.json')
DETAIL_DIR = os.path.join(HERE, '..', 'web', 'public', 'runs')

# the grid the lab lets you dial through
GRID = dict(
    n_layer=[1, 2, 4, 6],
    n_head=[1, 2, 4],
    n_embd=[32, 64, 128],
    block_size=[16, 32, 64, 128],
)

# fixed training recipe, shared by every run so results are comparable
DROPOUT_STUDY = dict(dropout=[0.1, 0.2], where=dict(n_embd=128, block_size=128))

RECIPE = dict(
    batch_size=32,
    max_iters=3000,
    eval_interval=150,
    eval_iters=40,
    learning_rate=1e-3,
    dropout=0.0,
    seed=1337,
)

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

with open(os.path.join(HERE, '..', '..', 'input.txt'), 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# a fixed held-out prompt from the validation split; each run records what the
# trained model actually predicts after it (and where its heads look)
_val_text = text[n:]
PROBE_TEXT = _val_text[:_val_text.find('Therefore, my lord') + len('Therefore, my lor')]
PROBE_TEXT_NEXT = 'd'


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.last_wei = wei.detach()  # read by measure.py's attention probe
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, n_layer, n_head, n_embd, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


def sync():
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def run_id(cfg):
    rid = f"L{cfg['n_layer']}_H{cfg['n_head']}_E{cfg['n_embd']}_T{cfg['block_size']}"
    d = cfg.get('dropout', 0.0)
    return rid + (f"_D{round(d * 100):02d}" if d else '')


def train(cfg, max_iters=None, quiet=False):
    r = dict(RECIPE)
    if max_iters is not None:
        r['max_iters'] = max_iters
    torch.manual_seed(r['seed'])
    block_size = cfg['block_size']

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (r['batch_size'],))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    dropout = cfg.get('dropout', r['dropout'])
    model = GPTLanguageModel(cfg['n_layer'], cfg['n_head'], cfg['n_embd'], block_size, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(r['eval_iters'])
            for k in range(r['eval_iters']):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    optimizer = torch.optim.AdamW(model.parameters(), lr=r['learning_rate'])
    curve = []
    train_seconds = 0.0
    wall_start = time.time()
    for it in range(r['max_iters'] + 1):
        if it % r['eval_interval'] == 0 or it == r['max_iters']:
            losses = estimate_loss()
            curve.append(dict(step=it, train=round(losses['train'], 4), val=round(losses['val'], 4)))
            if not quiet:
                print(f"  {run_id(cfg)} step {it}: train {losses['train']:.4f} val {losses['val']:.4f}", flush=True)
        if it == r['max_iters']:
            break
        t0 = time.time()
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        sync()
        train_seconds += time.time() - t0

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(dict(config=cfg, state_dict=model.state_dict()), os.path.join(CKPT_DIR, run_id(cfg) + '.pt'))

    return dict(
        id=run_id(cfg),
        config=cfg,
        n_params=n_params,
        curve=curve,
        final_train_loss=curve[-1]['train'],
        final_val_loss=curve[-1]['val'],
        best_val_loss=min(p['val'] for p in curve),
        train_seconds=round(train_seconds, 2),
        wall_seconds=round(time.time() - wall_start, 2),
        # tokens the optimizer consumed per second of (synchronized) training step time
        train_tok_s=round(r['batch_size'] * block_size * r['max_iters'] / train_seconds, 1),
    )


def grid_configs():
    keys = list(GRID.keys())
    for vals in itertools.product(*[GRID[k] for k in keys]):
        yield dict(zip(keys, vals))


def study_configs():
    # the largest-model corner overfits (train ~1.30 vs val ~1.55 at dropout 0), so retrain it
    # with dropout, everything else identical, to see whether regularization closes the gap
    for cfg in grid_configs():
        if all(cfg[k] == v for k, v in DROPOUT_STUDY['where'].items()):
            for d in DROPOUT_STUDY['dropout']:
                yield dict(cfg, dropout=d)


def all_configs():
    yield from grid_configs()
    yield from study_configs()


def aggregate():
    import measure
    runs = {}
    os.makedirs(DETAIL_DIR, exist_ok=True)
    for cfg in all_configs():
        rid = run_id(cfg)
        p = os.path.join(RUNS_DIR, rid + '.json')
        mp = measure.measure_path(rid)
        if not (os.path.exists(p) and os.path.exists(mp)):
            continue
        with open(p) as f:
            run = json.load(f)
        with open(mp) as f:
            m = json.load(f)
        run['config'].setdefault('dropout', RECIPE['dropout'])
        run['speed'] = dict(train_tok_s=run['train_tok_s'], **m['speed'])
        del run['train_tok_s']
        runs[rid] = run
        with open(os.path.join(DETAIL_DIR, rid + '.json'), 'w') as f:
            json.dump(m['detail'], f, separators=(',', ':'))
    meta = dict(
        grid=GRID,
        dropout_study=DROPOUT_STUDY,
        recipe=RECIPE,
        measure=measure.SETTINGS,
        dataset=dict(name='tinyshakespeare', chars=len(text), vocab_size=vocab_size,
                     train_tokens=len(train_data), val_tokens=len(val_data)),
        vocab=chars,
        hardware=dict(device=device, machine=platform.machine(), processor=platform.processor(),
                      torch=torch.__version__),
        n_runs=len(runs),
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(dict(meta=meta, runs=runs), f, separators=(',', ':'))
    print(f"wrote {len(runs)} runs -> {os.path.relpath(OUT_JSON)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=str, default=None)
    ap.add_argument('--aggregate', action='store_true')
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
        return

    import measure
    os.makedirs(RUNS_DIR, exist_ok=True)
    # cheapest first so the lab becomes usable early; the dropout study after the grid
    cfgs = sorted(grid_configs(), key=lambda c: c['n_layer'] * c['n_embd'] * c['block_size']) + list(study_configs())
    if args.only:
        cfgs = [c for c in cfgs if run_id(c) == args.only]
    for i, cfg in enumerate(cfgs):
        p = os.path.join(RUNS_DIR, run_id(cfg) + '.json')
        if os.path.exists(p) and os.path.exists(os.path.join(CKPT_DIR, run_id(cfg) + '.pt')):
            if not os.path.exists(measure.measure_path(run_id(cfg))):
                measure.measure(run_id(cfg))
            continue
        t0 = time.time()
        res = train(cfg, quiet=True)
        with open(p, 'w') as f:
            json.dump(res, f)
        # measure right away, on an otherwise idle GPU, so timings aren't contended
        m = measure.measure(run_id(cfg))
        print(f"[{i + 1}/{len(cfgs)}] {res['id']} params={res['n_params']} val={res['final_val_loss']:.4f} "
              f"train={res['train_seconds']:.1f}s ({res['train_tok_s']:.0f} tok/s) "
              f"prefill={m['speed']['prefill_tok_s']:.0f} tok/s gen={m['speed']['gen_tok_s']:.0f} tok/s "
              f"wall={time.time() - t0:.1f}s", flush=True)
        aggregate()  # keep runs.json current so an open lab picks up new runs
    aggregate()


if __name__ == '__main__':
    main()
