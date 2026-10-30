// Screen-space retro interfaces: the terminal console, component info cards, the printout and
// the lab notebook.

import { FB } from './fb'
import { P, mix } from './palette'
import { wrap, CHAR_W } from './font'
import type { Picker } from './pick'
import { Config, Param, PARAMS, Run, RunsFile, runId, paramCounts } from './data'
import type { Info } from './info'
import { Experiment, phaseOf, measuredSeconds, generatedCount, SLOW_TOK_S } from './experiment'

const fmt = (n: number) => n.toLocaleString('en-US')
const fmtK = (n: number) => (n >= 10000 ? `${(n / 1000).toFixed(0)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${Math.round(n)}`)

export function panel(fb: FB, x: number, y: number, w: number, h: number, edge = P.steel) {
  fb.rect(x, y, w, h, P.ink)
  fb.frame(x, y, w, h, edge)
  fb.rect(x + 1, y + 1, w - 2, 1, mix(edge, P.ink, 0.5))
}

export function button(fb: FB, picker: Picker, x: number, y: number, w: number, h: number, label: string, on: boolean, action: () => void, col = P.lime, hint?: string) {
  const id = picker.ui(action, hint)
  fb.rect(x, y, w, h, on ? col : P.moss, id)
  fb.frame(x, y, w, h, on ? mix(col, P.white, 0.4) : P.pine)
  fb.text(x + Math.round((w - label.length * CHAR_W + 1) / 2), y + Math.round((h - 7) / 2), label, on ? P.ink : col)
}

// ---------------------------------------------------------------- control terminal

export interface ControlState {
  cfg: Config
  row: number
  grid: Record<Param, number[]>
  run: Run | undefined
  nRuns: number
  totalRuns: number
  /** dropout rates that were recorded for the dialled architecture */
  dropouts: number[]
  stepDropout(dir: number): void
  /** dials and RUN are locked while the training replay / ingest is playing */
  locked: boolean
  realtime: boolean
  step(p: Param, dir: number): void
  setRow(i: number): void
  runExperiment(): void
  toggleSpeed(): void
  toggleScale(): void
}

export const CONTROL = { w: 196, h: 136 }

export function drawControl(fb: FB, picker: Picker, s: ControlState, t: number) {
  const w = CONTROL.w, h = CONTROL.h
  const x = 6, y = fb.h - h - 6
  // beige plastic bezel around a phosphor screen
  fb.rect(x, y, w, h, P.tan)
  fb.frame(x, y, w, h, P.rust)
  fb.rect(x + 1, y + 1, w - 2, 1, P.sand)
  fb.rect(x + 1, y + h - 2, w - 2, 1, P.clay)
  const sx = x + 5, sy = y + 5, sw = w - 10, sh = h - 15
  fb.rect(sx, sy, sw, sh, mix(P.moss, P.ink, 0.6))
  fb.frame(sx, sy, sw, sh, P.rust)
  fb.text(x + 6, y + h - 9, 'TGL-3000', P.rust)
  fb.rect(x + 56, y + h - 7, 5, 3, s.locked ? P.lime : P.pine)
  const sc = picker.ui(s.toggleScale, 'G: compare scale with GPT-3 / GPT-6.1')
  fb.rect(x + 66, y + h - 10, 35, 9, P.clay, sc)
  fb.frame(x + 66, y + h - 10, 35, 9, P.rust)
  fb.text(x + 69, y + h - 9, 'SCALE', P.sand)
  // generation playback speed: slowed down to watch, or at the measured per-token latency
  const spd = picker.ui(s.toggleSpeed, 'R: generation playback speed')
  const label = s.realtime ? 'GEN: REAL TIME' : `GEN: SLOW ${SLOW_TOK_S}/S`
  fb.rect(x + w - 92, y + h - 10, 87, 9, s.realtime ? P.amber : P.clay, spd)
  fb.frame(x + w - 92, y + h - 10, 87, 9, P.rust)
  fb.text(x + w - 90, y + h - 9, label, s.realtime ? P.ink : P.sand)

  let ty = sy + 4
  fb.text(sx + 5, ty, 'TINYGPT LAB CONTROL', P.lime)
  const tally = `${s.nRuns}/${s.totalRuns}`
  fb.text(sx + sw - 4 - tally.length * CHAR_W, ty, tally, P.green)
  ty += 12
  PARAMS.forEach((p, i) => {
    const active = i === s.row
    const rowY = ty + i * 12
    const rowPick = picker.ui(() => s.setRow(i))
    fb.rect(sx + 2, rowY - 2, sw - 4, 11, active ? P.pine : mix(P.moss, P.ink, 0.6), rowPick)
    fb.text(sx + 5, rowY, (active ? '>' : ' ') + p, active ? P.white : P.lime)
    const vals = s.grid[p], vi = vals.indexOf(s.cfg[p])
    const bx = sx + 86
    button(fb, picker, bx, rowY - 1, 9, 9, '◄', vi > 0, () => s.step(p, -1), P.lime, `${p} down`)
    const v = String(s.cfg[p])
    fb.text(bx + 12 + (18 - v.length * CHAR_W) / 2 + 3, rowY, v, P.white)
    button(fb, picker, bx + 36, rowY - 1, 9, 9, '►', vi < vals.length - 1, () => s.step(p, 1), P.lime, `${p} up`)
    for (let k = 0; k < vals.length; k++) {
      const dx = bx + 50 + k * 5
      if (k === vi) fb.rect(dx, rowY + 2, 3, 3, P.lemon)
      else fb.rect(dx + 1, rowY + 3, 1, 1, P.green)
    }
  })
  {
    // fifth row: training-time dropout, only adjustable where variants were recorded
    const i = PARAMS.length, rowY = ty + i * 12, active = s.row === i
    const vals = s.dropouts, d = s.cfg.dropout ?? 0, vi = vals.indexOf(d)
    fb.rect(sx + 2, rowY - 2, sw - 4, 11, active ? P.pine : mix(P.moss, P.ink, 0.6), picker.ui(() => s.setRow(i)))
    fb.text(sx + 5, rowY, (active ? '>' : ' ') + 'dropout', vals.length > 1 ? (active ? P.white : P.lime) : P.green)
    const bx = sx + 86
    button(fb, picker, bx, rowY - 1, 9, 9, '◄', vi > 0, () => s.stepDropout(-1), P.lime, 'dropout down')
    const v = d ? d.toFixed(1) : '0'
    fb.text(bx + 12 + (18 - v.length * CHAR_W) / 2 + 3, rowY, v, vals.length > 1 ? P.white : P.steel)
    button(fb, picker, bx + 36, rowY - 1, 9, 9, '►', vi < vals.length - 1, () => s.stepDropout(1), P.lime,
      vals.length > 1 ? 'dropout up' : 'dropout variants: n_embd 128 + block_size 128 only')
    for (let k = 0; k < vals.length; k++) {
      const dx = bx + 50 + k * 5
      if (k === vi) fb.rect(dx, rowY + 2, 3, 3, P.lemon)
      else fb.rect(dx + 1, rowY + 3, 1, 1, P.green)
    }
  }
  ty += (PARAMS.length + 1) * 12 + 3
  fb.rect(sx + 3, ty - 2, sw - 6, 1, P.pine)
  const params = paramCounts(s.cfg).total
  fb.text(sx + 5, ty + 1, 'PARAMS', P.green)
  fb.text(sx + 47, ty + 1, fmt(params), P.white)
  ty += 11
  if (s.run) fb.text(sx + 5, ty, `RECORD ${runId(s.cfg)}`, P.green)
  else fb.text(sx + 5, ty, (t * 2) % 2 < 1.4 ? 'NOT YET RECORDED' : '', P.amber)
  ty += 12
  const can = !!s.run && !s.locked
  button(fb, picker, sx + 5, ty, sw - 10, 12, s.locked ? 'RUNNING...' : 'RUN EXPERIMENT ↵', can, () => can && s.runExperiment(), P.lime)
  // scanlines over the screen
  for (let yy = sy + 1; yy < sy + sh - 1; yy += 2)
    for (let xx = sx + 1; xx < sx + sw - 1; xx++) {
      const i = yy * fb.w + xx
      if (i >= 0 && i < fb.color.length) fb.color[i] = mix(fb.color[i], P.ink, 0.22)
    }
}

// ---------------------------------------------------------------- component info popup

export function drawInfo(fb: FB, picker: Picker, info: Info, anchor: [number, number] | null, close: () => void) {
  const x = 6, y = 6, pad = 6
  const maxH = fb.h - CONTROL.h - 20
  const layout = (w: number) => {
    const tw = w - pad * 2
    const lines: [string, number, number][] = [] // text, colour, indent
    const push = (s: string, c: number, indent = 0) => lines.push([s, c, indent])
    for (const l of wrap(info.what, tw)) push(l, P.mist)
    if (info.io.length) {
      push('', 0)
      push('TENSORS', P.silver)
      for (const [k, v] of info.io) push(`${k.padEnd(12, ' ')}${v}`, P.cyan)
      push(`${'params'.padEnd(12, ' ')}${info.params}`, P.lemon)
    }
    if (info.effects.length) {
      push('', 0)
      push('ARCHITECTURE DIALS', P.silver)
      for (const [p, e] of info.effects) {
        wrap(e, tw - 66).forEach((l, i) => push(`${(i === 0 ? p : '').padEnd(11, ' ')}${l}`, P.amber))
      }
    }
    if (info.recorded) {
      push('', 0)
      push('FROM THE RECORDED RUN', P.silver)
      for (const r of info.recorded) for (const l of wrap(r, tw)) push(l, P.lime)
    }
    return lines
  }
  // widen the card rather than let it run under the console
  let w = 216, lines = layout(w)
  for (const cand of [264, 312]) {
    if (30 + lines.length * 9 + 4 <= maxH) break
    w = Math.min(cand, fb.w - (NOTEBOOK_W + 24))
    lines = layout(w)
  }
  const fit = Math.floor((maxH - 34) / 9)
  if (lines.length > fit) lines = lines.slice(0, fit)
  const h = 30 + lines.length * 9 + 4
  if (anchor) {
    const [ax, ay] = anchor
    const ex = x + w, ey = Math.min(y + h - 4, Math.max(y + 10, ay))
    fb.line(ex, ey, ax, ay, P.lemon)
    fb.rect(ax - 1, ay - 1, 3, 3, P.lemon)
  }
  panel(fb, x, y, w, h, P.lemon)
  fb.rect(x + 1, y + 1, w - 2, 21, P.night)
  fb.text(x + pad, y + 4, info.title, P.lemon)
  fb.text(x + pad, y + 13, info.tag, P.silver)
  button(fb, picker, x + w - 13, y + 3, 10, 10, '×', true, close, P.lemon, 'close')
  lines.forEach(([s, c], i) => fb.text(x + pad, y + 27 + i * 9, s, c))
}

// ---------------------------------------------------------------- printout

/** hard-wrap like a line printer: break at newlines or the column limit, keep char offsets */
function hardWrap(s: string, cols: number): { text: string; start: number }[] {
  const out: { text: string; start: number }[] = []
  let line = '', start = 0
  for (let i = 0; i < s.length; i++) {
    const ch = s[i]
    if (ch === '\n') { out.push({ text: line, start }); line = ''; start = i + 1; continue }
    if (line.length === cols) { out.push({ text: line, start }); line = ''; start = i }
    line += ch
  }
  out.push({ text: line, start })
  return out
}

/**
 * The run report the terminal prints. It feeds up out of a printer slot at the foot of the
 * sidebar (cabled to the terminal) so it never covers the machine. Sections appear as the
 * experiment reaches them; the generated text streams onto the paper token by token.
 */
export function drawPrintout(fb: FB, picker: Picker, e: Experiment, now: number, meta: RunsFile['meta'],
  realtime: boolean, terminal: [number, number], area: { x: number; w: number; top: number }, close: () => void) {
  const W = area.w
  const run = e.run, d = e.detail, sp = run.speed
  const { phase, k } = phaseOf(e, now)
  const ink = P.ink, faint = mix(P.sand, P.clay, 0.45), soft = mix(P.ink, P.slate, 0.35)
  const cols = Math.floor((W - 18) / CHAR_W)

  const items: { h: number; draw: (x: number, y: number) => void }[] = []
  const line = (s: string, col = ink) => {
    for (const l of wrap(s, cols * CHAR_W)) items.push({ h: 9, draw: (x, y) => fb.text(x, y, l, col) })
  }
  const rule = (title: string) => line(`-- ${title} ${'-'.repeat(Math.max(0, cols - title.length - 4))}`, faint)
  const gap = (h = 4) => items.push({ h, draw: () => {} })

  line('TGL-3000   RUN REPORT')
  line(`${run.id}  ${fmtK(run.n_params)} PARAMS`, soft)
  gap()
  rule('TRAINING')
  const tk = phase === 'train' ? k : 1
  items.push({
    h: 44, draw: (x, y) => {
      // dot-matrix loss curve: val solid, train dotted
      const cw = W - 22, ch = 38
      const all = run.curve.flatMap(p => [p.train, p.val])
      const lo = Math.min(...all), hi = Math.max(...all), last = run.curve[run.curve.length - 1].step
      for (let xx = 0; xx < cw; xx += 2) fb.px(x + xx, y + ch, faint)
      for (let yy = 0; yy < ch; yy += 2) fb.px(x, y + yy, faint)
      for (const key of ['train', 'val'] as const) {
        for (let xx = 0; xx < cw; xx += key === 'val' ? 1 : 2) {
          const st = (xx / cw) * last
          if (st > tk * last) break
          const i = Math.min(run.curve.length - 2, Math.floor((st / last) * (run.curve.length - 1)))
          const a = run.curve[i], b = run.curve[i + 1]
          const v = a[key] + (b[key] - a[key]) * ((st - a.step) / (b.step - a.step))
          fb.px(x + xx, y + ch - ((v - lo) / (hi - lo)) * ch, key === 'val' ? ink : P.clay)
        }
      }
    },
  })
  const shown = run.curve[Math.min(run.curve.length - 1, Math.floor(tk * (run.curve.length - 1)))]
  line(phase === 'train' ? `STEP ${shown.step}  VAL ${shown.val.toFixed(3)}` : `VAL ${shown.val.toFixed(3)}  TRAIN ${shown.train.toFixed(3)}`)
  if (phase !== 'train') {
    if (e.prev) {
      const dv = run.final_val_loss - e.prev.final_val_loss
      line(`${dv <= 0 ? 'BETTER' : 'WORSE'} BY ${Math.abs(dv).toFixed(3)} VS ${e.prev.id}`, dv <= 0 ? P.pine : P.wine)
    }
    const dr = run.config.dropout ?? 0
    line(`DROPOUT ${dr ? dr.toFixed(1) : '0 (OFF)'}  LR ${meta.recipe.learning_rate}`, dr ? P.wine : soft)
    line(`${meta.recipe.max_iters} STEPS IN ${run.train_seconds.toFixed(1)} S`)
    line(`${fmtK(sp.train_tok_s)} TOK/S  (BATCH ${meta.recipe.batch_size}x${run.config.block_size})`)
    gap()
    rule('INGEST')
    line(`${d.prompt.length}-CHAR PROMPT, ONE PASS`)
    line(`${sp.prefill_ms.toFixed(2)} MS = ${fmtK(sp.prefill_tok_s)} TOK/S`)
  }
  if (phase === 'generate' || phase === 'done') {
    gap()
    rule('GENERATE')
    line(`${sp.gen_ms_per_tok.toFixed(2)} MS/TOK = ${fmtK(sp.gen_tok_s)} TOK/S`)
    const n = generatedCount(e)
    line(`${n}/${d.steps.length} TOKENS  ${measuredSeconds(e).toFixed(3)} S`, soft)
    if (phase === 'generate') line(realtime ? '>> AT MEASURED SPEED' : `>> SLOWED TO ${SLOW_TOK_S} TOK/S`, P.wine)
    gap(3)
    // the prompt's tail in faint ink, then the model's own characters as they arrive
    const tail = d.prompt.slice(-Math.min(d.prompt.length, cols))
    const body = tail + d.text.slice(0, n)
    for (const l of hardWrap(body, cols)) {
      items.push({
        h: 9, draw: (x, y) => {
          for (let i = 0; i < l.text.length; i++) fb.text(x + i * CHAR_W, y, l.text[i], l.start + i < tail.length ? faint : ink)
        },
      })
    }
    if (phase === 'done') { gap(3); line(e.stopped ? '-- STOPPED --' : '-- END OF RUN --', faint) }
  }
  gap(6)

  // feed the slip up out of the printer slot; once it reaches the notebook, older lines scroll off
  const [tx, ty] = terminal
  const slot = fb.h - 12
  const total = items.reduce((a, it) => a + it.h, 0)
  const visH = Math.max(0, Math.min(total, slot - area.top))
  const top = slot - visH
  const px = area.x
  const tear = picker.ui(close, 'tear off the printout')
  fb.rect(px, top, W, visH, P.sand, tear)
  for (let y = top + 3; y < slot - 2; y += 6) {
    fb.rect(px + 2, y, 3, 3, mix(P.ink, P.night, 0.4))
    fb.rect(px + W - 5, y, 3, 3, mix(P.ink, P.night, 0.4))
  }
  fb.rect(px + 7, top, 1, visH, faint)
  fb.rect(px + W - 8, top, 1, visH, faint)
  let y = slot - total
  for (const it of items) {
    if (y >= top + 2 && y + it.h <= slot) it.draw(px + 10, y)
    y += it.h
  }
  for (let x = 0; x < W; x++) if ((x * 7) % 5 < 2) fb.px(px + x, top, mix(P.sand, P.ink, 0.5))
  fb.rect(px + W, top + 2, 1, visH - 2, P.ink)
  // the printer: a slot in a beige housing, cabled back to the terminal's CRT
  fb.line(px - 3, slot + 4, tx, ty, P.rust)
  fb.rect(px - 4, slot, W + 8, 7, P.tan)
  fb.frame(px - 4, slot, W + 8, 7, P.rust)
  fb.rect(px + 2, slot + 1, W - 4, 2, P.ink)
  fb.rect(px + W - 6, slot + 4, 3, 2, phase === 'generate' ? P.lime : P.pine)
}

// ---------------------------------------------------------------- lab notebook

export const NOTEBOOK_W = 204

/** every run replayed this session, ranked by validation loss, plus a params-vs-loss scatter */
export function drawNotebook(fb: FB, picker: Picker, history: Run[], current: string | null, busy: boolean, now: number,
  dial: (r: Run) => void, close: () => void): { x: number; bottom: number } {
  const W = NOTEBOOK_W, x = fb.w - W - 6, y = 6
  const rows = [...history].sort((a, b) => a.final_val_loss - b.final_val_loss).slice(0, 12)
  const h = 30 + rows.length * 10 + 6 + 84 + 4
  panel(fb, x, y, W, h, P.lemon)
  fb.rect(x + 1, y + 1, W - 2, 12, P.night)
  fb.text(x + 5, y + 3, 'LAB NOTEBOOK', P.lemon)
  fb.text(x + 85, y + 3, `${history.length} run${history.length === 1 ? '' : 's'}`, P.silver)
  button(fb, picker, x + W - 12, y + 2, 10, 10, '×', true, close, P.lemon, 'close')

  let ty = y + 17
  fb.text(x + 5, ty, '#  L H   E   T DR   VAL GEN/S', P.steel)
  ty += 11
  rows.forEach((run, i) => {
    const cur = run.id === current
    const id = picker.ui(() => dial(run), `dial in ${run.id}`)
    fb.rect(x + 2, ty - 1, W - 4, 10, cur ? P.night : P.ink, id)
    const c = run.config
    const val = cur && busy ? ' ... ' : run.final_val_loss.toFixed(3)
    const col = cur ? (busy && (now * 4) % 2 < 1 ? P.amber : P.lemon) : P.mist
    const dr = c.dropout ? `.${Math.round(c.dropout * 10)}` : ' 0'
    fb.text(x + 5, ty, `${String(i + 1).padEnd(2)} ${c.n_layer} ${c.n_head} ${String(c.n_embd).padStart(3)} ${String(c.block_size).padStart(3)} ${dr} ${val} ${fmtK(run.speed.gen_tok_s).padStart(4)}`, col)
    ty += 10
  })

  // scatter: parameters (log scale) against final validation loss; lower is better
  ty += 6
  const sx = x + 18, sw = W - 26, sh = 66, sy = ty
  fb.rect(sx, sy, sw, sh, mix(P.night, P.ink, 0.5))
  fb.frame(sx - 1, sy - 1, sw + 2, sh + 2, P.slate)
  const lp0 = Math.log10(1.5e4), lp1 = Math.log10(1.5e6)
  const vals = history.map(r => r.final_val_loss)
  const v0 = Math.min(1.4, ...vals) - 0.05, v1 = Math.max(2.2, ...vals) + 0.05
  for (const p of [1e5, 1e6]) {
    const gx = sx + ((Math.log10(p) - lp0) / (lp1 - lp0)) * sw
    for (let yy = sy; yy < sy + sh; yy += 2) fb.px(gx, yy, P.slate)
  }
  for (const run of history) {
    const cur = run.id === current
    if (cur && busy) continue
    const px = sx + ((Math.log10(run.n_params) - lp0) / (lp1 - lp0)) * sw
    const py = sy + ((run.final_val_loss - v0) / (v1 - v0)) * sh
    fb.rect(px - (cur ? 2 : 1), py - (cur ? 2 : 1), cur ? 5 : 3, cur ? 5 : 3, cur ? P.lemon : P.silver)
  }
  microNum(fb, x + 4, sy + 1, v0.toFixed(1), P.steel)
  microNum(fb, x + 4, sy + sh - 6, v1.toFixed(1), P.steel)
  fb.text(sx, sy + sh + 3, 'params  100k   1M     val', P.steel)
  return { x, bottom: y + h }
}

/** 3x5 digits with a one-pixel decimal point */
function microNum(fb: FB, x: number, y: number, s: string, col: number) {
  for (const ch of s) {
    if (ch === '.') { fb.px(x, y + 4, col); x += 2; continue }
    fb.micro(x, y, ch, col)
    x += 4
  }
}

// ---------------------------------------------------------------- little bits

export function drawTooltip(fb: FB, mx: number, my: number, s: string) {
  const w = s.length * CHAR_W + 5
  let x = Math.round(mx + 8), y = Math.round(my + 10)
  if (x + w > fb.w - 2) x = Math.round(mx - w - 4)
  if (y + 12 > fb.h - 2) y = Math.round(my - 14)
  fb.rect(x, y, w, 11, P.ink)
  fb.frame(x, y, w, 11, P.silver)
  fb.text(x + 3, y + 2, s, P.white)
}

export function drawHint(fb: FB, s: string, y: number, right = 0) {
  const w = s.length * CHAR_W
  const x = Math.round((fb.w - right - w) / 2)
  fb.text(x + 1, y + 1, s, P.ink)
  fb.text(x, y, s, P.silver)
}
