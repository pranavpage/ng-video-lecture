// The experiment timeline: replay the recorded training run, ingest the held-out prompt, then
// generate token by token from the recorded trace — slowed down, or at the measured latencies.

import type { Run, RunDetail } from './data'
import type { Reveal } from './scene'

export const TRAIN_S = 4.1 // training replay (the loss curve prints)
export const INGEST_S = 1.6 // prompt cartridges feed in, one forward pass
export const SLOW_TOK_S = 10 // slow-motion generation rate

export type Phase = 'train' | 'ingest' | 'generate' | 'done'

export interface Experiment {
  run: Run
  detail: RunDetail
  prev: Run | null
  start: number
  /** tokens generated so far; the fraction is progress through the current token */
  genPos: number
  stopped: boolean
}

export function phaseOf(e: Experiment, now: number): { phase: Phase; k: number } {
  const el = now - e.start
  if (el < TRAIN_S) return { phase: 'train', k: el / TRAIN_S }
  if (el < TRAIN_S + INGEST_S) return { phase: 'ingest', k: (el - TRAIN_S) / INGEST_S }
  if (e.genPos < e.detail.steps.length && !e.stopped) return { phase: 'generate', k: e.genPos / e.detail.steps.length }
  return { phase: 'done', k: 1 }
}

/** move generation forward by dt seconds of playback */
export function advance(e: Experiment, now: number, dt: number, realtime: boolean) {
  if (phaseOf(e, now).phase !== 'generate') return
  const steps = e.detail.steps
  let budget = dt
  while (budget > 0 && e.genPos < steps.length) {
    const i = Math.floor(e.genPos)
    const dur = realtime ? steps[i][1] / 1000 : 1 / SLOW_TOK_S
    const left = (i + 1 - e.genPos) * dur
    if (budget >= left) { e.genPos = i + 1; budget -= left } else { e.genPos += budget / dur; budget = 0 }
  }
}

/** measured wall-clock time of the tokens generated so far */
export function measuredSeconds(e: Experiment): number {
  let ms = 0
  const n = Math.floor(e.genPos)
  for (let i = 0; i < n; i++) ms += e.detail.steps[i][1]
  return ms / 1000
}

export function generatedCount(e: Experiment): number {
  return Math.min(e.detail.steps.length, Math.floor(e.genPos))
}

function topkProbs(step: RunDetail['steps'][number], vocab: number): number[] {
  const out = new Array(vocab).fill(0)
  for (const [i, p] of step[3]) out[i] = p
  return out
}

export interface MachineState {
  reveal: Reveal
  front: number | null
  carts: number | null
  feedback: number | null
  busy: boolean
  /** dropout rate to visualize (training replay only; dropout is off when generating) */
  dropout: number
}

const IDLE: MachineState = { reveal: { context: null, output: null, probs: null, chosen: -1, attn: null }, front: null, carts: null, feedback: null, busy: false, dropout: 0 }

/** what the machine should be doing right now for this experiment */
export function machineState(e: Experiment | null, now: number, vocab: number): MachineState {
  if (!e) return IDLE
  const { phase, k } = phaseOf(e, now)
  const d = e.detail, T = d.prompt.length
  const el = now - e.start
  switch (phase) {
    case 'train':
      // repeated sweeps up the stack: the optimizer stepping through batches
      return { ...IDLE, front: (el * 1.3) % 1, busy: true, dropout: e.run.config.dropout ?? 0 }
    case 'ingest':
      return {
        reveal: { context: d.prompt, output: null, probs: k > 0.97 ? d.probs : null, chosen: -1, attn: k > 0.85 ? d.attn_last : null },
        carts: Math.min(1, k / 0.55),
        front: k > 0.55 ? (k - 0.55) / 0.42 : null,
        feedback: null,
        busy: true,
        dropout: 0,
      }
    case 'generate': {
      const i = Math.floor(e.genPos), pulse = e.genPos - i
      const step = d.steps[i]
      const sampled = pulse > 0.55
      return {
        reveal: {
          context: (d.prompt + d.text.slice(0, i)).slice(-T),
          // once sampled, the new character drops onto the exit belt (and then loops back)
          output: d.text.slice(0, i + (sampled ? 1 : 0)),
          probs: topkProbs(step, vocab),
          chosen: sampled ? step[0] : -1,
          attn: d.attn_last,
        },
        carts: null,
        front: sampled ? null : pulse / 0.55,
        feedback: sampled ? (pulse - 0.55) / 0.45 : null,
        busy: true,
        dropout: 0,
      }
    }
    case 'done': {
      const n = generatedCount(e)
      const last = d.steps[Math.max(0, n - 1)]
      return {
        reveal: { context: (d.prompt + d.text.slice(0, n)).slice(-T), output: d.text.slice(0, n), probs: n ? topkProbs(last, vocab) : d.probs, chosen: -1, attn: d.attn_last },
        carts: null, front: null, feedback: null, busy: false, dropout: 0,
      }
    }
  }
}
