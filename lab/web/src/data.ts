// Recorded training runs, produced by lab/train/sweep.py. Nothing here is synthesized.

export interface Config {
  n_layer: number; n_head: number; n_embd: number; block_size: number
  /** training-time dropout; 0 everywhere except the dropout study */
  dropout?: number
}
/** the four architecture dials */
export type Param = 'n_layer' | 'n_head' | 'n_embd' | 'block_size'

export interface CurvePoint { step: number; train: number; val: number }

export interface Speed {
  /** tokens consumed per second of (synchronized) optimizer-step time, batch 32 */
  train_tok_s: number
  /** one forward pass over the whole prompt, batch 1 (median of repeats) */
  prefill_ms: number
  prefill_tok_s: number
  /** median per-token latency while generating, batch 1, no KV cache */
  gen_ms_per_tok: number
  gen_tok_s: number
  gen_total_s: number
}

/** summary of one recorded run (small; all of them load up front) */
export interface Run {
  id: string
  config: Config
  n_params: number
  curve: CurvePoint[]
  final_train_loss: number
  final_val_loss: number
  best_val_loss: number
  train_seconds: number
  wall_seconds: number
  speed: Speed
}

/** [sampled vocab index, latency ms, p(sampled), [[vocab index, p], ...top-k]] */
export type GenStep = [number, number, number, [number, number][]]

/** what the trained model did with the held-out prompt (fetched on demand per run) */
export interface RunDetail {
  id: string
  prompt: string
  target: string
  probs: number[]
  attn_last: number[][][] // [layer][head][position]
  text: string
  steps: GenStep[]
}

export interface RunsFile {
  meta: {
    grid: Record<Param, number[]>
    /** the corner of the grid that was also trained with dropout */
    dropout_study?: { dropout: number[]; where: Partial<Record<Param, number>> }
    recipe: {
      batch_size: number; max_iters: number; eval_interval: number; eval_iters: number
      learning_rate: number; dropout: number; seed: number
    }
    measure: { gen_tokens: number; topk: number; prefill_repeats: number; batch: number; method: string }
    dataset: { name: string; chars: number; vocab_size: number; train_tokens: number; val_tokens: number }
    vocab: string[]
    hardware: { device: string; machine: string; processor: string; torch: string }
    n_runs: number
  }
  runs: Record<string, Run>
}

export const PARAMS: Param[] = ['n_layer', 'n_head', 'n_embd', 'block_size']

export function runId(c: Config): string {
  const d = c.dropout ?? 0
  return `L${c.n_layer}_H${c.n_head}_E${c.n_embd}_T${c.block_size}` + (d ? `_D${String(Math.round(d * 100)).padStart(2, '0')}` : '')
}

/** dropout rates recorded for this architecture (always includes 0) */
export function dropoutOptions(c: Config, meta: RunsFile['meta'] | undefined): number[] {
  const st = meta?.dropout_study
  if (!st) return [0]
  const inStudy = (Object.entries(st.where) as [Param, number][]).every(([k, v]) => c[k] === v)
  return inStudy ? [0, ...st.dropout] : [0]
}

export async function loadRuns(): Promise<RunsFile> {
  const res = await fetch(`${import.meta.env.BASE_URL}runs.json`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`runs.json: HTTP ${res.status}`)
  return res.json()
}

const details = new Map<string, Promise<RunDetail>>()
export function loadDetail(id: string): Promise<RunDetail> {
  let p = details.get(id)
  if (!p) {
    p = fetch(`${import.meta.env.BASE_URL}runs/${id}.json`).then(r => {
      if (!r.ok) throw new Error(`runs/${id}.json: HTTP ${r.status}`)
      return r.json()
    })
    p.catch(() => details.delete(id))
    details.set(id, p)
  }
  return p
}

// exact parameter counts for the gpt.py architecture (they match n_params in every run file)
export const VOCAB = 65
export function paramCounts(c: Config) {
  const C = c.n_embd, T = c.block_size
  const head = 3 * C * (C / c.n_head)
  const proj = C * C + C
  const ffn = 8 * C * C + 5 * C
  const ln = 2 * C
  const block = c.n_head * head + proj + ffn + 2 * ln
  return {
    tok: VOCAB * C,
    pos: T * C,
    head,
    proj,
    ffn,
    ln,
    block,
    lm: VOCAB * C + VOCAB,
    total: VOCAB * C + T * C + c.n_layer * block + ln + VOCAB * C + VOCAB,
  }
}
