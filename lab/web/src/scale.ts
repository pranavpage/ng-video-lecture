// Scale chart: how the dialled-in machine compares with gpt.py's full model, GPT-3 and GPT-6.1.
// Only published figures are used; anything a lab has not disclosed is shown as undisclosed.

import { FB } from './fb'
import { P, mix } from './palette'
import { wrap, CHAR_W } from './font'
import type { Picker } from './pick'
import { Config, paramCounts } from './data'
import { panel, button } from './ui'

interface Model {
  name: string
  params: number | null
  layers: number | null
  width: number | null
  context: number | null
  /** tokens seen during training */
  tokens: number | null
  colour: number
}

// gpt.py's defaults in this repo: n_layer 6, n_head 6, n_embd 384, block_size 256,
// 5000 iterations of batch 64 x 256 characters
const GPT_PY: Config = { n_layer: 6, n_head: 6, n_embd: 384, block_size: 256 }

// Brown et al. 2020, "Language Models are Few-Shot Learners", Table 2.1 / section 2.2
const GPT3: Model = { name: 'GPT-3', params: 175e9, layers: 96, width: 12288, context: 2048, tokens: 300e9, colour: P.cyan }

// OpenAI API docs (Oct 2026): 1,050,000-token context window. Size and architecture undisclosed.
const GPT61: Model = { name: 'GPT-6.1', params: null, layers: null, width: null, context: 1_050_000, tokens: null, colour: P.magenta }

export const SCALE_SOURCES = 'n/a = not disclosed. Sources: Brown et al. 2020 (GPT-3); OpenAI API docs, Oct 2026 (GPT-6.1).'

function lab(c: Config, steps: number, batch: number): Model {
  return {
    name: 'THIS MACHINE', params: paramCounts(c).total, layers: c.n_layer, width: c.n_embd,
    context: c.block_size, tokens: steps * batch * c.block_size, colour: P.lemon,
  }
}

const gptPy: Model = {
  name: 'gpt.py full', params: paramCounts(GPT_PY).total, layers: 6, width: 384, context: 256,
  tokens: 5000 * 64 * 256, colour: P.lime,
}

function short(n: number): string {
  const u: [number, string][] = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'k']]
  for (const [v, s] of u) if (n >= v) return `${+(n / v).toPrecision(3)}${s}`
  return `${n}`
}

function times(a: number, b: number): string {
  const r = a / b
  return r >= 100 ? Math.round(r).toLocaleString('en-US') : r.toPrecision(2)
}

export function drawScale(fb: FB, picker: Picker, c: Config, steps: number, batch: number, close: () => void) {
  const me = lab(c, steps, batch)
  const models = [me, gptPy, GPT3, GPT61]
  const metrics: { label: string; key: keyof Model; max: number }[] = [
    { label: 'PARAMETERS', key: 'params', max: 1e12 },
    { label: 'LAYERS', key: 'layers', max: 1e3 },
    { label: 'WIDTH (d_model)', key: 'width', max: 1e5 },
    { label: 'CONTEXT WINDOW (tokens)', key: 'context', max: 1e7 },
    { label: 'TOKENS SEEN IN TRAINING', key: 'tokens', max: 1e13 },
  ]
  const W = Math.min(350, fb.w - 12), x = 6, y = 6, pad = 6
  const labelW = 13 * CHAR_W, valW = 8 * CHAR_W
  const barX = x + pad + labelW, barW = W - pad * 2 - labelW - valW - 4

  const notes = [
    `GPT-3: ${times(GPT3.params!, me.params!)}x the parameters, ${times(GPT3.layers!, me.layers!)}x the layers, ${times(GPT3.context!, me.context!)}x the context.`,
    `GPT-6.1: ${times(GPT61.context!, me.context!)}x the context. At this lab's one-storey-per-block scale, GPT-3 would stand ${times(96, c.n_layer)}x as tall.`,
    'GPT-3/6.1 tokens are subwords (a few chars); this lab\'s are single characters.',
  ].flatMap(n => wrap(n, W - pad * 2))
  const srcLines = wrap(SCALE_SOURCES, W - pad * 2)
  const h = 26 + metrics.length * (11 + models.length * 9) + 6 + notes.length * 9 + 4 + srcLines.length * 9 + 4

  panel(fb, x, y, W, h, P.cyan)
  fb.rect(x + 1, y + 1, W - 2, 21, P.night)
  fb.text(x + pad, y + 4, 'SCALE CHART', P.cyan)
  fb.text(x + pad, y + 13, 'log scale: each tick is 10x', P.silver)
  button(fb, picker, x + W - 13, y + 3, 10, 10, '×', true, close, P.cyan, 'close')

  let ty = y + 27
  for (const m of metrics) {
    fb.text(x + pad, ty, m.label, P.silver)
    ty += 10
    const decades = Math.log10(m.max)
    for (const mod of models) {
      const v = mod[m.key] as number | null
      fb.text(x + pad, ty, mod.name, mod === me ? P.lemon : P.mist)
      // decade ticks
      for (let d = 0; d <= decades; d++) fb.px(barX + Math.round((d / decades) * barW), ty + 7, P.slate)
      if (v === null) {
        // undisclosed: hatched band across the whole range plus a label
        for (let i = 0; i < barW; i += 3) fb.px(barX + i, ty + 3, mix(mod.colour, P.ink, 0.5))
        fb.text(barX + barW + 4, ty, 'n/a', P.steel)
      } else {
        const len = Math.max(1, Math.round((Math.log10(Math.max(1, v)) / decades) * barW))
        fb.rect(barX, ty + 1, len, 5, mod.colour)
        fb.text(barX + barW + 4, ty, short(v), P.white)
      }
      ty += 9
    }
    ty += 1
  }
  ty += 4
  for (const n of notes) { fb.text(x + pad, ty, n, P.mist); ty += 9 }
  ty += 4
  for (const s of srcLines) { fb.text(x + pad, ty, s, P.steel); ty += 9 }
}
