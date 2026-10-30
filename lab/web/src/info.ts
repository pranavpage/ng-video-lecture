// Contextual explanations for each clickable component of the machine.

import { Config, paramCounts, Run, RunDetail, VOCAB } from './data'

export interface Info {
  title: string
  tag: string // short material/kind label
  what: string
  io: [string, string][] // label, shape
  params: string
  effects: [string, string][] // parameter, how it changes this component
  recorded?: string[] // lines drawn from the recorded run (only after it has been run)
}

const fmt = (n: number) => n.toLocaleString('en-US')
const fmtK = (n: number) => (n >= 10000 ? `${(n / 1000).toFixed(1)}k` : fmt(Math.round(n)))
const show = (ch: string) => (ch === '\n' ? '↵' : ch === ' ' ? '·' : ch)

export interface Recorded { run: Run; detail: RunDetail }

export function describe(comp: string, c: Config, rec: Recorded | null, vocab: string[]): Info | null {
  const info = describeBase(comp, c, rec, vocab)
  // dropout sits after the attention weights, the attention projection and the feedforward
  if (info && /\.(head\d+|proj|ffn)$/.test(comp)) {
    const d = c.dropout ?? 0
    info.effects.push(['dropout', d
      ? `zeroes a random ${Math.round(d * 100)}% of its outputs at each training step (never when generating)`
      : 'off (0) for this run'])
  }
  return info
}

function describeBase(comp: string, c: Config, rec: Recorded | null, vocab: string[]): Info | null {
  const run = rec?.detail ?? null, speed = rec?.run.speed
  const C = c.n_embd, T = c.block_size, H = c.n_head, hs = C / H
  const pc = paramCounts(c)
  const BTC = `(B, ${T}, ${C})`
  const m = comp.match(/^b(\d+)\.(.+)$/)
  const layer = m ? Number(m[1]) : -1
  const rest = m ? m[2] : comp
  const sub = rest.startsWith('head') ? 'head' : rest
  const headIdx = rest.startsWith('head') ? Number(rest.slice(4)) : -1
  const blk = `BLOCK ${layer + 1}/${c.n_layer}`

  switch (sub) {
    case 'tok':
      return {
        title: 'TOKEN EMBEDDING', tag: 'nn.Embedding(65, n_embd)',
        what: `Each glowing cartridge is one character of the context. The intake looks up a learned vector for every character in a table with one row per vocabulary character (${VOCAB}) and n_embd columns.`,
        io: [['in  idx', `(B, ${T}) ints`], ['out tok_emb', BTC]],
        params: `${VOCAB} x ${C} = ${fmt(pc.tok)}`,
        effects: [['block_size', `${T} cartridges on the conveyor`], ['n_embd', `each vector is ${C} wide`]],
        recorded: run && speed ? [
          `Cartridges hold the last ${T} chars of a held-out prompt ending "${run.prompt.slice(-12).replace(/\n/g, '↵')}", then each generated char as it is fed back.`,
          `Ingesting the whole prompt (one forward pass, batch 1) took ${speed.prefill_ms.toFixed(2)} ms: ${fmtK(speed.prefill_tok_s)} tok/s.`,
        ] : undefined,
      }
    case 'pos':
      return {
        title: 'POSITIONAL EMBEDDING', tag: 'nn.Embedding(block_size, n_embd)',
        what: 'A second lookup table indexed by position 0..T-1. Its vectors are added to the token vectors, so the same character carries different information at different places in the context.',
        io: [['in  arange(T)', `(${T},)`], ['out pos_emb', `(${T}, ${C}) + tok_emb`], ['=> x', BTC]],
        params: `${T} x ${C} = ${fmt(pc.pos)}`,
        effects: [['block_size', `${T} numbered marks on the strip`], ['n_embd', `${C}-wide position vectors`]],
      }
    case 'ln1': case 'ln2': case 'lnf':
      return {
        title: sub === 'lnf' ? 'FINAL LAYER NORM  ln_f' : `LAYER NORM  ${sub}  ${blk}`, tag: 'nn.LayerNorm(n_embd)',
        what: (sub === 'ln1' ? 'Normalizes every token vector before it enters self-attention. '
          : sub === 'ln2' ? 'Normalizes every token vector before it enters the feedforward stage. '
            : 'Normalizes the final residual stream before the output projection. ')
          + 'Each vector is shifted to zero mean and unit variance, then scaled by a learned gain and bias.',
        io: [['in', BTC], ['out', BTC]],
        params: `2 x ${C} = ${fmt(pc.ln)}`,
        effects: [['n_embd', `normalizes ${C} features per token`]],
      }
    case 'head': {
      const h = headIdx
      const info: Info = {
        title: `ATTENTION HEAD ${h + 1}/${H}  ${blk}`, tag: 'Head(head_size)',
        what: 'Every position emits a query, a key and a value. Query . key scores decide how much each earlier character matters; the lit triangle is the causal mask, so position t can only read positions 0..t. Output is the score-weighted sum of values.',
        io: [['in', BTC], ['q, k, v', `(B, ${T}, ${hs})`], ['weights', `(B, ${T}, ${T})`], ['out', `(B, ${T}, ${hs})`]],
        params: `3 x ${C} x ${hs} = ${fmt(pc.head)}`,
        effects: [
          ['n_head', `${H} parallel heads, each ${hs} wide (n_embd / n_head)`],
          ['block_size', `${T} x ${T} attention grid`],
          ['n_embd', 'wider projections'],
        ],
      }
      if (run) {
        const w = run.attn_last[layer]?.[h]
        if (w) {
          let best = 0
          for (let i = 1; i < w.length; i++) if (w[i] > w[best]) best = i
          const ctx = run.prompt
          info.recorded = [
            `Lights: where the last position of the prompt looked in this run${T > 8 ? ` (${T} positions summed into 8 cells)` : ''}.`,
            `Strongest: "${show(ctx[best])}" at position ${best} (${(w[best] * 100).toFixed(0)}%), ${T - 1 - best === 0 ? "the last char itself" : `${T - 1 - best} char${T - 1 - best === 1 ? "" : "s"} back`}.`,
          ]
        }
      }
      return info
    }
    case 'proj':
      return {
        title: `ATTENTION OUTPUT PROJECTION  ${blk}`, tag: 'nn.Linear(n_embd, n_embd)',
        what: 'The circuit plate that joins the heads: their outputs are concatenated side by side, then mixed back into the residual stream by one linear layer.',
        io: [['in concat', `(B, ${T}, ${H} x ${hs})`], ['out', BTC]],
        params: `${C} x ${C} + ${C} = ${fmt(pc.proj)}`,
        effects: [['n_head', `joins ${H} head outputs`], ['n_embd', 'grows with n_embd squared']],
      }
    case 'ffn':
      return {
        title: `FEEDFORWARD  ${blk}`, tag: 'Linear -> ReLU -> Linear',
        what: 'Heavy per-position processing, applied to each token independently. It expands to 4 x n_embd (the four bays), applies ReLU, then projects back. Attention gathers information; this is where the block computes with it.',
        io: [['in', BTC], ['hidden', `(B, ${T}, ${4 * C})`], ['out', BTC]],
        params: `8 x ${C}^2 + 5 x ${C} = ${fmt(pc.ffn)}`,
        effects: [['n_embd', `${4 * C} hidden units; params grow with n_embd squared`]],
      }
    case 'res1': case 'res2':
      return {
        title: `RESIDUAL CONNECTION  ${blk}`, tag: 'x = x + f(x)',
        what: `This conduit carries x around the ${sub === 'res1' ? 'layer norm + self-attention' : 'layer norm + feedforward'} stage and adds it back at the "+" junction. Information can pass straight through the stack, which keeps deep models trainable.`,
        io: [['carries', BTC], ['adds', `${sub === 'res1' ? 'attention' : 'feedforward'} output ${BTC}`]],
        params: '0 (just addition)',
        effects: [['n_layer', `${2 * c.n_layer} conduits for ${c.n_layer} block${c.n_layer > 1 ? 's' : ''}`], ['n_embd', `stream is ${C} wide`]],
      }
    case 'lm': {
      const info: Info = {
        title: 'OUTPUT PROJECTION  lm_head', tag: 'nn.Linear(n_embd, 65)',
        what: `Maps every position's vector to ${VOCAB} logits, one per character, and softmax turns them into next-character probabilities. Each terminal on top is one character of the vocabulary.`,
        io: [['in', BTC], ['out logits', `(B, ${T}, ${VOCAB})`]],
        params: `${C} x ${VOCAB} + ${VOCAB} = ${fmt(pc.lm)}`,
        effects: [['n_embd', `${C} inputs per logit`]],
      }
      if (run && speed) {
        const order = run.probs.map((p, i) => [p, i] as [number, number]).sort((a, b) => b[0] - a[0]).slice(0, 4)
        const ti = vocab.indexOf(run.target)
        info.recorded = [
          `After "...${run.prompt.slice(-14).replace(/\n/g, '↵')}" this model predicts:`,
          order.map(([p, i]) => `"${show(vocab[i])}" ${(p * 100).toFixed(1)}%`).join('  '),
          `True next char "${run.target}": ${(run.probs[ti] * 100).toFixed(1)}%`,
          `Generating: ${speed.gen_ms_per_tok.toFixed(2)} ms per token (${fmtK(speed.gen_tok_s)} tok/s). Each token reruns all ${T} positions, so one token costs about as much as ingesting the whole prompt.`,
        ]
      }
      return info
    }
    case 'out':
      return {
        title: 'OUTPUT STREAM', tag: 'torch.multinomial(softmax(logits[:, -1]))',
        what: 'Generation happens here. The last position\'s 65 probabilities are sampled to pick one character, which leaves on this belt. A copy loops back onto the input conveyor as the newest context token, and the whole machine runs again for the next character. That loop is autoregression.',
        io: [['in logits', `(B, ${T}, ${VOCAB}) -> last position`], ['sample', '(B, 1)'], ['context', `cat -> crop to last ${T}`]],
        params: '0 (sampling)',
        effects: [['block_size', `the loop keeps only the last ${T} chars`]],
        recorded: run && speed ? [
          `${run.steps.length} characters were generated continuing the prompt.`,
          `Each took a median ${speed.gen_ms_per_tok.toFixed(2)} ms (${fmtK(speed.gen_tok_s)} tok/s), a full pass over ${T} positions every time, since gpt.py has no KV cache.`,
        ] : undefined,
      }
    case 'terminal':
      return {
        title: 'CONTROL TERMINAL', tag: 'the researcher\'s console',
        what: 'Dial in n_layer, n_head, n_embd and block_size, then run the experiment. Every combination on these dials was actually trained ahead of time; running replays that recorded run.',
        io: [], params: '-', effects: [],
      }
  }
  return null
}
