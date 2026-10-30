# TinyGPT Lab

A pixel-art isometric laboratory where the GPT from `../gpt.py` hovers over the floor as a
giant machine. Dial `n_layer`, `n_head`, `n_embd` and `block_size` at the terminal and the
machine reassembles itself. Run an experiment and the machine replays the recorded training
run, ingests a held-out prompt, then generates text token by token from the recorded trace —
slowed down to watch, or at the latency that was actually measured (`R` toggles).

```
lab/
  train/sweep.py   trains every config on the dial grid, saves train/ckpt/<id>.pt, then measures it
  train/measure.py prefill + generation benchmarks and the token trace, from a checkpoint
                   -> web/public/runs.json (summary) + web/public/runs/<id>.json (detail)
  web/             the lab (Vite + TypeScript, custom software rasterizer, no runtime deps)
```

## Run it

```bash
cd lab/web && npm install && npm run dev      # http://localhost:5173
```

## Regenerate the recorded runs

```bash
cd lab
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python torch numpy
cd train && ../.venv/bin/python sweep.py      # resumable; refreshes runs.json after every run
../.venv/bin/python measure.py --force        # re-measure every checkpoint without retraining
```

The grid is `n_layer ∈ {1,2,4,6} × n_head ∈ {1,2,4} × n_embd ∈ {32,64,128} × block_size ∈ {16,32,64,128}`
(144 runs). Every run uses the same recipe: 3000 steps, batch 32, AdamW lr 1e-3, dropout 0, seed 1337,
tinyshakespeare with the same 90/10 split as `gpt.py`. The architecture is `gpt.py`'s, unchanged.

**Dropout study.** Dropout is 0 across the grid (`gpt.py` uses 0.2). The largest corner
(`n_embd` 128, `block_size` 128) starts to overfit, so its 12 configs are also trained at
dropout 0.1 and 0.2, everything else identical — 24 extra runs, 168 in total, ids suffixed
`_D10` / `_D20`. The console's fifth row selects them; it is fixed at 0 elsewhere. During the
training replay the machine blinks that fraction of attention / feedforward cells dark;
generation always runs with dropout off (eval mode), as in `gpt.py`.

## What is real

Everything the lab reports comes from the run files:

- **training**: loss curves (train/val every 150 steps), final losses, parameter count, the
  summed time of the synchronized optimizer steps, and tokens/s derived from it
- **ingestion**: one forward pass over the whole held-out prompt (the last `block_size`
  characters of validation text ending `"Therefore, my lor"`), batch 1, median of 30 timed
  repeats after warm-up
- **generation**: 300 tokens continuing that prompt exactly like `gpt.py`'s `generate()`
  (batch 1, no KV cache, a full-context forward per token, fixed-seed sampling). Every step
  records the sampled character, its top-8 candidates and that step's own wall-clock latency.
- **probe**: the next-character distribution after the prompt and every head's attention
  for its last position

On the machine, input enters bottom-left and output leaves top-right: the pink cartridges
are the real context window, the output terminals light with that step's recorded top-8
probabilities, the sampled character drops onto the brass exit belt on the far side, and a
copy rides the return loop back onto the input conveyor (autoregression). Slow motion runs
at 10 tok/s and says so; real-time playback uses the recorded per-token latencies. The printout prints the measured numbers either way.

Because there is no KV cache, generating one token costs about as much as ingesting the
whole prompt — the printout's ingest and generate lines make that visible.

Tensor shapes and parameter counts in the info cards are computed from the config and match
`n_params` in every run. Configs that haven't been trained yet show `NOT YET RECORDED`.

## Scale chart

`G` (or SCALE on the console) compares the dialled-in machine with `gpt.py`'s full model,
GPT-3 and GPT-6.1 on log-scale bars: parameters, layers, width, context window, training
tokens. GPT-3's figures are from Brown et al. 2020 (table 2.1); for GPT-6.1 only the
context window (1,050,000 tokens, OpenAI API docs, Oct 2026) is public, so everything else is
shown as not disclosed rather than estimated. GPT-3/6.1 count subword tokens, this lab
counts characters.

## Results UI

A prototype compared four results interfaces (side log, ticker strip, in-world printout,
lab notebook); the printout + notebook combination won. The variants live on the throwaway
branch `prototype/results-ui`.

## Controls

Arrow keys / WASD dial the parameters, Enter runs, `R` toggles slow-motion / real-time
generation, Esc closes a card. Click a notebook row to dial that experiment back in. Click
any part of the machine for an explanation; hover the output terminals or cartridges for
per-character detail.
