// The floating transformer: animated parts, per-component materials, and the run replay.

import { FB, Mode, scale } from './fb'
import { P, RAMPS, mix, Ramp } from './palette'
import { box, bayer, clamp01, hash, nextGroup, smooth, FaceShader } from './draw3d'
import { Box, Layout, Part } from './layout'
import type { Picker } from './pick'
import { glyph } from './font'

interface Live {
  p: Part
  cur: Box
  vel: Box
  born: number
  dying: number | null
  dir: [number, number]
  landed: boolean
}

interface Spark { x: number; y: number; z: number; vx: number; vy: number; vz: number; life: number; c: number }

/** what the machine is showing from a recorded run; every field comes from the run's files */
export interface Reveal {
  /** characters on the conveyor, one per cartridge (length block_size) */
  context: string | null
  /** generated characters so far; the newest sits at the exit end of the output belt */
  output: string | null
  /** per-vocabulary probabilities lighting the output terminals */
  probs: number[] | null
  /** vocabulary index that was just sampled (-1 = none) */
  chosen: number
  attn: number[][][] | null
}

export interface DrawCtx {
  t: number
  picker: Picker
  selected: string | null
  hovered: string | null
  reveal: Reveal
  /** 0..1 progress of a signal sweeping up the stack (null = idle) */
  front: number | null
  /** 0..1 progress of the cartridges being fed into the intake */
  carts: number | null
  /** 0..1 progress of the sampled token travelling back to the conveyor */
  feedback: number | null
  /** an experiment is running: conduits flow fast, terminals flicker */
  busy: boolean
  /** dropout rate during the training replay: that fraction of cells blinks dark each step */
  dropout: number
  vocab: string[]
}

const BOX_KEYS: (keyof Box)[] = ['x', 'y', 'z', 'w', 'd', 'h']
const copy = (b: Box): Box => ({ ...b })
const zero = (): Box => ({ x: 0, y: 0, z: 0, w: 0, d: 0, h: 0 })

export class Scene {
  live = new Map<string, Live>()
  sparks: Spark[] = []
  layout!: Layout
  shake = 0

  setLayout(layout: Layout, now: number) {
    const first = !this.layout
    this.layout = layout
    const seen = new Set<string>()
    for (const p of layout.parts) {
      seen.add(p.key)
      const l = this.live.get(p.key)
      if (l && l.dying === null) { l.p = p; continue }
      let cur: Box
      const src = p.spawnFrom ? this.live.get(p.spawnFrom) : undefined
      if (first) cur = { ...p.box, z: p.box.z + 260 + (p.delay ?? 0) * 200 }
      else if (src && src.dying === null) cur = copy(src.cur)
      else cur = { ...p.box, z: p.box.z + 70 }
      this.live.set(p.key, { p, cur, vel: zero(), born: now, dying: null, dir: [0, 0], landed: false })
    }
    for (const [k, l] of this.live) {
      if (seen.has(k) || l.dying !== null) continue
      l.dying = now
      const cx = l.cur.x + l.cur.w / 2, cy = l.cur.y + l.cur.d / 2
      const n = Math.hypot(cx, cy) || 1
      l.dir = [cx / n, cy / n]
      l.vel.z += 40 + hash(k.length * 31) * 40
      l.vel.x += l.dir[0] * 60
      l.vel.y += l.dir[1] * 60
    }
    if (!first) this.shake = 1.5
  }

  update(dt: number, now: number) {
    dt = Math.min(dt, 1 / 20)
    const K = 130, C = 15
    for (const [k, l] of this.live) {
      if (l.dying !== null) {
        l.cur.x += l.vel.x * dt; l.cur.y += l.vel.y * dt; l.cur.z += l.vel.z * dt
        l.vel.z -= 220 * dt
        l.cur.w *= 1 - dt * 1.5; l.cur.d *= 1 - dt * 1.5; l.cur.h *= 1 - dt * 1.5
        if (now - l.dying > 0.7) this.live.delete(k)
        continue
      }
      const age = now - l.born - (l.p.delay ?? 0)
      if (age < 0) continue
      for (const f of BOX_KEYS) {
        const target = l.p.box[f]
        l.vel[f] += ((target - l.cur[f]) * K - l.vel[f] * C) * dt
        l.cur[f] += l.vel[f] * dt
      }
      if (!l.landed && Math.abs(l.cur.z - l.p.box.z) < 1.5 && age > 0.1) {
        l.landed = true
        if (l.p.kind !== 'cart' && l.p.kind !== 'post') this.burst(l.cur, l.p.kind === 'head' ? P.cyan : l.p.kind === 'ffn' ? P.amber : P.mist, 6)
      }
    }
    for (const s of this.sparks) {
      s.x += s.vx * dt; s.y += s.vy * dt; s.z += s.vz * dt
      s.vz -= 160 * dt
      s.life -= dt
    }
    this.sparks = this.sparks.filter(s => s.life > 0)
    this.shake = Math.max(0, this.shake - dt * 4)
  }

  burst(b: Box, c: number, n: number) {
    if (this.sparks.length > 600) return
    for (let i = 0; i < n; i++) {
      const edge = Math.random()
      this.sparks.push({
        x: b.x + (i & 1 ? b.w : edge * b.w), y: b.y + (i & 1 ? edge * b.d : b.d), z: b.z + Math.random() * b.h,
        vx: (Math.random() - 0.3) * 40, vy: (Math.random() - 0.3) * 40, vz: 20 + Math.random() * 50,
        life: 0.3 + Math.random() * 0.4, c,
      })
    }
  }

  draw(fb: FB, ctx: DrawCtx) {
    const bob = Math.sin(ctx.t * 1.3) * 1.6
    const shakeX = this.shake > 0 ? (Math.random() - 0.5) * this.shake : 0
    const L = this.layout

    // floor shadow beneath the hovering machine
    const sw = L.W + 16, sd = L.D + 16
    fb.face([-sw / 2, -sd / 2, 0.01], [sw, 0, 0], [0, sd, 0], (a, b) => {
      const nx = (a / sw) * 2 - 1, ny = (b / sd) * 2 - 1
      const r = Math.max(Math.abs(nx), Math.abs(ny)) * 0.75 + Math.hypot(nx, ny) * 0.25
      return r < 0.92 - bob * 0.01 ? 1 : 0
    }, 0, 0, Mode.Shadow)
    const cw = L.lanes * 8 + 4
    fb.face([-cw / 2, L.D / 2, 0.01], [cw, 0, 0], [0, 122, 0], (a, b) => (a > 1 && a < cw - 1 && b < 120 ? 1 : 0), 0, 0, Mode.Shadow)

    const front = ctx.front === null ? -1e9 : L.hover + smooth(ctx.front) * (L.top - L.hover + 10)
    const glowQueue: (() => void)[] = []

    for (const l of this.live.values()) {
      const p = l.p
      const age = ctx.t - l.born - (p.delay ?? 0)
      let vis = l.dying !== null ? 1 - (ctx.t - l.dying) / 0.7 : clamp01(age / 0.35)
      if (vis <= 0) continue
      const b: Box = { ...l.cur, z: l.cur.z + bob, x: l.cur.x + shakeX }
      const pick = ctx.picker.world(p.comp, p.kind === 'post' || p.kind === 'cart' || p.kind === 'outcart' ? p.index ?? -1 : -1)
      const isSel = !!p.comp && (p.comp === ctx.selected || p.comp === ctx.hovered)
      // activation from the replay's signal front
      let act = 0
      if (p.kind === 'cart' || p.kind === 'rail') {
        if (ctx.carts !== null) act = p.kind === 'rail' ? 1 : p.index! < ctx.carts * L.T ? 1 - ctx.carts * 0.6 : 0
        // the newest cartridge flashes as a sampled token lands on it
        if (p.kind === 'cart' && ctx.feedback !== null && p.index === L.T - 1) act = Math.max(act, ctx.feedback)
      } else if (p.kind === 'outcart' || p.kind === 'outrail') {
        // the freshly sampled character glows as it exits
        if (p.kind === 'outcart' && p.index === 0 && ctx.reveal.chosen >= 0) act = 1 - (ctx.feedback ?? 0) * 0.6
      } else if (ctx.front !== null) {
        const zc0 = l.p.box.z, zc1 = l.p.box.z + l.p.box.h
        const dist = front < zc0 ? zc0 - front : front > zc1 ? front - zc1 : 0
        act = Math.max(0, 1 - dist / 18)
      }
      if (isSel) act = Math.max(act, 0.35 + 0.15 * Math.sin(ctx.t * 6))
      let sh = shaderFor(p, ctx, act, l, fb.zoom, L)
      if (ctx.dropout > 0 && (p.kind === 'head' || p.kind === 'ffn' || p.kind === 'proj')) {
        // dropout: each training step zeroes a random fraction p of activations
        const inner = sh, rate = ctx.dropout, stepN = Math.floor(ctx.t * 8), seed = l.p.key.length * 131 + (p.layer ?? 0) * 977 + (p.head ?? 0) * 59
        sh = (f, a, b, fw, fh, px, py) => {
          const c = inner(f, a, b, fw, fh, px, py)
          if (c === 0 || a < 1 || b < 1 || a > fw - 1 || b > fh - 1) return c
          const cell = Math.floor(a / 3) * 31 + Math.floor(b / 3) * 17 + f * 7 + seed
          return hash(cell * 7919 + stepN * 104729) < rate ? mix(c, P.ink, 0.8) : c
        }
      }
      box(fb, b, sh, pick, nextGroup(), vis)
      queueGlows(glowQueue, fb, p, b, ctx, act)
    }

    // autoregression: a copy of the character leaving on the exit belt loops around the front of
    // the machine and lands on the input conveyor as the newest context token
    const from = this.live.get('out0'), to = this.live.get(`cart${L.T - 1}`)
    if (ctx.reveal.output !== null && from && to && from.dying === null && to.dying === null) {
      const a = from.cur, c = to.cur
      const p0 = [a.x + a.w / 2, a.y + a.d / 2, a.z + a.h + bob + 1]
      const p2 = [c.x + c.w / 2, c.y + c.d / 2, c.z + c.h + bob + 1]
      const p1 = [L.W / 2 + 60, L.D / 2 + 60, Math.max(p0[2], p2[2]) + 10]
      const q = (u: number) => [0, 1, 2].map(j => (1 - u) * (1 - u) * p0[j] + 2 * (1 - u) * u * p1[j] + u * u * p2[j])
      // the return path, faintly marked while the machine is generating
      if (ctx.busy) {
        const drift = (ctx.t * 0.6) % (1 / 28)
        glowQueue.push(() => {
          let k = 0
          for (let u = drift; u < 1; u += 1 / 28, k++) { const [x, y, z] = q(u); fb.glow(x, y, z, scale(P.amber, k % 2 ? 0.45 : 0.75), k % 2 ? 0 : 1) }
        })
      }
      // the sampled character itself, riding the loop as a brass capsule
      if (ctx.feedback !== null && ctx.reveal.chosen >= 0) {
        const u = smooth(ctx.feedback)
        const [x, y, z] = q(u)
        box(fb, { x: x - 2.5, y: y - 2.5, z: z - 2, w: 5, d: 5, h: 4 }, (f, aa, bb, fw, fh) =>
          f === 0 && aa > 0.8 && bb > 0.8 && aa < fw - 0.8 && bb < fh - 0.8 ? P.white : base(RAMPS.brass, f, aa, bb, fw, fh, 0.8), 0, nextGroup())
        glowQueue.push(() => {
          for (let i = 1; i < 8; i++) {
            const [tx, ty, tz] = q(Math.max(0, u - i * 0.03))
            fb.glow(tx, ty, tz, scale(P.lemon, 1 - i / 8), i < 3 ? 1 : 0)
          }
        })
      }
    }

    this.glowQueue = glowQueue
    this.bobNow = bob
  }

  glowQueue: (() => void)[] = []
  bobNow = 0

  /** additive lights and sparks, drawn after the outline pass */
  drawGlows(fb: FB) {
    for (const g of this.glowQueue) g()
    for (const s of this.sparks) fb.glow(s.x, s.y, s.z, scale(s.c, clamp01(s.life * 3)), 0)
  }

  /** screen-space overlays: glyphs on cartridges, the top prediction tag */
  drawOverlays(fb: FB, ctx: DrawCtx) {
    const bob = this.bobNow
    const showGlyphs = fb.zoom >= 0.75
    for (const l of this.live.values()) {
      if (l.dying !== null || ctx.t - l.born - (l.p.delay ?? 0) < 0.35) continue
      const p = l.p, b = l.cur
      if (p.kind === 'cart' && showGlyphs && ctx.reveal.context) {
        const ch = ctx.reveal.context[p.index!] ?? ' '
        const sx = fb.sx(b.x + b.w / 2, b.y + b.d / 2), sy = fb.sy(b.x + b.w / 2, b.y + b.d / 2, b.z + b.h + bob)
        drawGlyphCentered(fb, ch === "\n" ? "↵" : ch, sx, sy, P.ink)
      }
      if (p.kind === 'outcart' && showGlyphs && ctx.reveal.output) {
        const out = ctx.reveal.output
        const ch = out[out.length - 1 - p.index!]
        if (ch === undefined) continue
        const sx = fb.sx(b.x + b.w / 2, b.y + b.d / 2), sy = fb.sy(b.x + b.w / 2, b.y + b.d / 2, b.z + b.h + bob)
        drawGlyphCentered(fb, ch === "\n" ? "↵" : ch, sx, sy, P.ink)
      }
    }
    // tag the sampled (or else the strongest) output terminal with its character and probability
    const probs = ctx.reveal.probs
    if (probs) {
      let best = 0
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i
      if (ctx.reveal.chosen >= 0) best = ctx.reveal.chosen
      const l = this.live.get(`post${best}`)
      if (l && l.dying === null) {
        const b = l.cur
        const sx = fb.sx(b.x + b.w / 2, b.y + b.d / 2), sy = fb.sy(b.x + b.w / 2, b.y + b.d / 2, b.z + b.h + bob)
        const ch = ctx.vocab[best]
        const label = `'${ch === '\n' ? '↵' : ch === ' ' ? '·' : ch}' ${(probs[best] * 100).toFixed(0)}%`
        const w = label.length * 6 + 5
        const tx = Math.round(sx - w / 2), ty = Math.round(sy - 26 + Math.sin(ctx.t * 2) * 1.5)
        fb.line(sx, sy - 3, sx, ty + 12, P.lemon)
        fb.rect(tx, ty, w, 12, P.ink)
        fb.frame(tx, ty, w, 12, P.lemon)
        fb.text(tx + 3, ty + 2, label, P.lemon)
      }
    }
  }

  /** world bounds of the target layout (for the camera) */
  bounds(): Box {
    const L = this.layout
    let x0 = Infinity, y0 = Infinity, z0 = 0, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity
    for (const p of L.parts) {
      const b = p.box
      x0 = Math.min(x0, b.x); y0 = Math.min(y0, b.y)
      x1 = Math.max(x1, b.x + b.w); y1 = Math.max(y1, b.y + b.d); z1 = Math.max(z1, b.z + b.h)
    }
    return { x: x0, y: y0, z: z0, w: x1 - x0, d: y1 - y0, h: z1 - z0 + 24 }
  }
}

function drawGlyphCentered(fb: FB, ch: string, sx: number, sy: number, col: number) {
  const g = glyph(ch)
  const ox = Math.round(sx - 2), oy = Math.round(sy - 4)
  for (const [x, y] of g) fb.px(ox + x, oy + y, col)
}

// ---------------------------------------------------------------- materials

function lit(c: number, act: number): number {
  return act > 0 ? mix(c, P.white, act * 0.55) : c
}

function base(ramp: Ramp, f: number, a: number, b: number, fw: number, fh: number, e: number): number {
  if (f === 0) return (a < e || b < e || a > fw - e || b > fh - e) ? ramp[0] : ramp[1]
  if (b > fh - e) return ramp[0]
  if (b < e) return ramp[3]
  return f === 1 ? ramp[2] : ramp[3]
}

/** sum attention weights into n buckets (when the context is longer than the display) */
export function bucket(w: number[], n: number): number[] {
  const out = new Array(n).fill(0)
  for (let i = 0; i < w.length; i++) out[Math.min(n - 1, Math.floor((i * n) / w.length))] += w[i]
  return out
}

function probColor(p: number): number {
  if (p < 0.005) return P.night
  if (p < 0.02) return P.wine
  if (p < 0.06) return P.red
  if (p < 0.15) return P.orange
  if (p < 0.35) return P.amber
  if (p < 0.6) return P.lemon
  return P.white
}

function shaderFor(p: Part, ctx: DrawCtx, act: number, l: Live, zoom: number, L: Layout): FaceShader {
  const e = Math.max(0.6, 1 / zoom)
  const t = ctx.t
  switch (p.kind) {
    case 'intake': {
      const slotW = L.lanes * 8 + 2
      return (f, a, b, fw, fh) => {
        if (f === 0) {
          // the embedding lookup table: a bank of 65 rows of memory cells
          if (a < 2 || b < 2 || a > fw - 2 || b > fh - 2) return lit(a < e || b < e ? P.steel : P.slate, act)
          const ca = Math.floor((a - 2) / 3.5), cb = Math.floor((b - 2) / 3.5)
          if ((a - 2) % 3.5 < 0.8 || (b - 2) % 3.5 < 0.8) return P.ink
          const n = hash(ca * 131 + cb * 7)
          return lit(n > 0.66 ? P.magenta : n > 0.33 ? P.grape : mix(P.grape, P.ink, 0.5), act)
        }
        if (f === 2) {
          const sa = (fw - slotW) / 2
          if (a > sa && a < sa + slotW && b > 1 && b < 11) {
            if (a < sa + 1 || a > sa + slotW - 1 || b < 2 || b > 10) return lit(P.pink, act)
            return P.ink
          }
        }
        if (b > 12 && b < 13.5) return lit(P.magenta, act + 0.2 * Math.sin(t * 3))
        return lit(base(RAMPS.darkMetal, f, a, b, fw, fh, e), act * 0.5)
      }
    }
    case 'rail': {
      const scroll = ctx.carts !== null || ctx.busy ? ctx.t * 30 : 0
      return (f, a, b, fw, fh) => {
        if (f === 0) return ((b + scroll) % 3 < 1) ? P.slate : P.night
        return base(RAMPS.darkMetal, f, a, b, fw, fh, e)
      }
    }
    case 'cart': {
      const glow = act
      return (f, a, b, fw, fh) => {
        if (f === 0 && a > 0.8 && b > 0.8 && a < fw - 0.8 && b < fh - 0.8) return lit(ctx.reveal.context ? P.peach : P.pink, glow)
        return lit(base(RAMPS.cartridge, f, a, b, fw, fh, e), glow)
      }
    }
    case 'pos': {
      const T = L.T
      return (f, a, b, fw, fh) => {
        if (f === 0) {
          if (a < e || b < e || a > fw - e || b > fh - e) return P.lemon
          // coordinate grid, one column per position
          const col = Math.floor((a / fw) * T), row = Math.floor((b / fh) * 4)
          if (((a / fw) * T) % 1 < 0.25) return P.clay
          return lit(((col + row) & 1) ? P.amber : P.orange, act)
        }
        if (b < 1.5 || b > fh - 1.5) return lit(f === 1 ? P.orange : P.clay, act * 0.5)
        // numbered strip: one illuminated tick per position, every 8th is a major mark
        const u = (a / fw) * T
        const i = Math.floor(u), fr = u - i
        const major = i % 8 === 0
        if (fr > 0.2 && fr < 0.8 && (major ? b < fh - 2 : b < fh * 0.6 && b > 2.5)) return lit(major ? P.white : P.lemon, act)
        return lit(f === 1 ? P.clay : P.rust, act * 0.6)
      }
    }
    case 'ln': {
      const pulse = 0.5 + 0.5 * Math.sin(t * 2.4 + (p.layer ?? 7) * 0.8)
      return (f, a, b, fw, fh) => {
        if (f === 0) {
          if (a < 1.2 || b < 1.2 || a > fw - 1.2 || b > fh - 1.2) return lit(P.mist, act)
          return lit(P.silver, act)
        }
        if (b > fh * 0.33 && b < fh * 0.67) return lit(mix(P.sea, P.cyan, pulse), act)
        return lit(b > fh * 0.67 ? P.mist : f === 1 ? P.steel : P.slate, act)
      }
    }
    case 'head': {
      const T = L.T
      const N = Math.min(T, 8)
      let row: number[] | null = null
      const attn = ctx.reveal.attn
      if (attn && p.layer! < attn.length) {
        const w = attn[p.layer!][p.head!]
        if (w) { row = bucket(w, N); const m = Math.max(...row) || 1; row = row.map(v => v / m) }
      }
      return (f, a, b, fw, fh) => {
        if (f === 2) {
          // causal mask: lower-triangular grid of query (row) x key (column) cells
          const m = 1.5
          if (a > m && a < fw - m && b > m && b < fh - m) {
            const cw = (fw - 2 * m) / N, ch = (fh - 2 * m) / N
            const col = Math.floor((a - m) / cw), r = Math.floor((fh - m - b) / ch)
            if ((a - m) % cw < Math.min(0.6, cw * 0.25) || (fh - m - b) % ch < Math.min(0.6, ch * 0.25)) return P.deepSea
            if (col > r) return mix(P.deepSea, P.ink, 0.5)
            if (r === N - 1 && row) return lit(mix(P.deepSea, P.white, row[col]), act * 0.4)
            return lit((col + r) & 1 ? P.sea : mix(P.sea, P.cyan, 0.4), act)
          }
        }
        if (f === 0 && (a < e || b < e || a > fw - e || b > fh - e)) return P.white
        // crystalline facets
        const u = a + b, v = a - b + 100
        if (u % 7 < 0.7 || v % 7 < 0.7) return lit(f === 0 ? P.sea : P.deepSea, act)
        const facet = (Math.floor(u / 7) + Math.floor(v / 7)) & 1
        const c = f === 0 ? (facet ? P.cyan : mix(P.cyan, P.white, 0.35)) : f === 1 ? (facet ? P.sea : P.deepSea) : (facet ? P.sea : mix(P.sea, P.deepSea, 0.5))
        return lit(c, act)
      }
    }
    case 'proj':
      return (f, a, b, fw, fh) => {
        if (f === 0) {
          const trace = (a % 6 < 0.7 && b % 12 > 2) || (b % 5 < 0.7 && a % 18 > 4)
          if (trace) return lit(P.cyan, act)
          if (a < e || b < e || a > fw - e || b > fh - e) return P.silver
          return lit(P.night, act * 0.5)
        }
        if (b > fh - e) return P.silver
        return lit((a % 4 < 1) ? P.sea : f === 1 ? P.slate : P.night, act * 0.6)
      }
    case 'ffn': {
      const vent = 0.5 + 0.5 * Math.sin(t * 5 + (p.layer ?? 0))
      return (f, a, b, fw, fh) => {
        if (f === 0) {
          // hazard-striped rim and riveted heavy plating
          if (a < 3 || b < 3 || a > fw - 3 || b > fh - 3) return ((Math.floor((a + b) / 3) & 1) ? P.amber : P.ink)
          const ra = (a - 3) % 8, rb = (b - 3) % 8
          if (ra > 3.4 && ra < 4.6 && rb > 3.4 && rb < 4.6) return P.tan
          if (ra < 0.6 || rb < 0.6) return P.rust
          return lit(P.copper, act * 0.5)
        }
        if (b > fh - 1.2) return P.tan
        if (b < 1.5) return P.plum
        // four bays: the 4 x n_embd hidden layer
        const bay = fw / 4, ba = a % bay
        if (ba < 1.6 || ba > bay - 0.4) return f === 1 ? P.rust : P.plum
        if ((ba < 3 || ba > bay - 2) && (b % 6 > 2.4 && b % 6 < 3.6)) return P.tan
        // vent slats glow while the block is computing
        if (b > 3 && b < fh - 3 && ba > 4 && ba < bay - 3) {
          if (b % 3 < 1.2) return P.plum
          return act > 0.05 ? mix(P.brick, P.amber, act * vent) : f === 1 ? P.brick : P.rust
        }
        return f === 1 ? P.brick : mix(P.brick, P.rust, 0.6)
      }
    }
    case 'pipe': {
      const speed = ctx.busy ? 60 : 10
      const vertical = p.box.h > p.box.w && p.box.h > p.box.d
      return (f, a, b, fw, fh) => {
        const along = vertical ? (f === 0 ? 0 : b) : (f === 0 ? (p.box.w > p.box.d ? a : b) : a)
        const dash = (((along - t * speed) % 9) + 9) % 9 < 2
        if (dash && f !== 0) return lit(act > 0 ? P.white : P.lime, act)
        return lit(base(RAMPS.conduit, f, a, b, fw, fh, e), act * 0.6)
      }
    }
    case 'adder':
      return (f, a, b, fw, fh) => {
        const ca = Math.abs(a - fw / 2), cb = Math.abs(b - fh / 2)
        if (f !== 0 && ((ca < 0.8 && cb < 2.2) || (cb < 0.8 && ca < 2.2))) return lit(P.lemon, act)
        return lit(base([P.lime, P.lime, P.green, P.pine], f, a, b, fw, fh, e), act * 0.6)
      }
    case 'lm':
      return (f, a, b, fw, fh) => {
        // exit chute on the +x face, where the output belt leaves the machine
        if (f === 1 && Math.abs(a - fw / 2) < 6 && b > 1.5 && b < 9.5) {
          if (Math.abs(a - fw / 2) > 5 || b < 2.5 || b > 8.5) return lit(P.amber, ctx.reveal.chosen >= 0 ? 0.8 : act)
          return P.ink
        }
        if (f === 0) {
          if (a < e || b < e || a > fw - e || b > fh - e) return P.steel
          return P.ink
        }
        // front panel: a row of small readout screens
        if (b > 4 && b < fh - 3) {
          const s = a % 7
          if (s > 1 && s < 6) return lit(ctx.reveal.probs ? mix(P.deepSea, P.sea, hash(Math.floor(a / 7) + Math.floor(t * 3)) * 0.8) : P.night, act)
        }
        return lit(base(RAMPS.darkMetal, f, a, b, fw, fh, e), act * 0.5)
      }
    case 'post': {
      let c = P.night
      const probs = ctx.reveal.probs
      if (probs) c = p.index === ctx.reveal.chosen ? P.white : probColor(probs[p.index!] ?? 0)
      else if (ctx.busy && act > 0.2) c = hash(p.index! + Math.floor(t * 12)) > 0.5 ? P.wine : P.night
      return (f, a, b, fw, fh) => {
        if (f === 0) return c
        if (b > fh - 1) return c === P.night ? P.slate : mix(c, P.ink, 0.3)
        return base(RAMPS.darkMetal, f, a, b, fw, fh, e)
      }
    }
    case 'outrail': {
      const scroll = ctx.busy ? -ctx.t * 30 : 0
      return (f, a, b, fw, fh) => {
        if (f === 0) return (((a + scroll) % 3) + 3) % 3 < 1 ? P.slate : P.night
        return base(RAMPS.darkMetal, f, a, b, fw, fh, e)
      }
    }
    case 'outcart': {
      const filled = !!ctx.reveal.output && p.index! < ctx.reveal.output.length
      return (f, a, b, fw, fh) => {
        if (!filled) return base(RAMPS.darkMetal, f, a, b, fw, fh, e)
        if (f === 0 && a > 0.8 && b > 0.8 && a < fw - 0.8 && b < fh - 0.8) return lit(P.lemon, act)
        return lit(base(RAMPS.brass, f, a, b, fw, fh, e), act)
      }
    }
    case 'emitter':
      return (f, a, b, fw, fh) => b < 1.2 && f !== 0 ? mix(P.sea, P.cyan, 0.5 + 0.5 * Math.sin(t * 8)) : base(RAMPS.darkMetal, f, a, b, fw, fh, e)
  }
}

// ---------------------------------------------------------------- lights

function queueGlows(q: (() => void)[], fb: FB, p: Part, b: Box, ctx: DrawCtx, act: number) {
  const t = ctx.t
  if (p.kind === 'emitter') {
    q.push(() => {
      for (let z = b.z - 2; z > 0; z -= 3) {
        const k = 0.25 + 0.25 * Math.sin(t * 10 + z * 0.3)
        if (bayer(Math.floor(z), Math.floor(t * 20)) > 0.6) continue
        fb.glow(b.x + b.w / 2, b.y + b.d / 2, z, scale(P.deepSea, k), 0)
      }
      fb.glow(b.x + b.w / 2, b.y + b.d / 2, b.z - 1, scale(P.cyan, 0.6), 1)
    })
  }
  if (p.kind === 'head' && ctx.reveal.attn) {
    const w = ctx.reveal.attn[p.layer!]?.[p.head!]
    if (!w) return
    const T = w.length, N = Math.min(T, 8)
    const row = bucket(w, N), m = Math.max(...row) || 1
    const boost = ctx.front !== null ? 1 + act : 0.8
    q.push(() => {
      // light signals hop along the bottom (last-query) row of the causal grid on the head's
      // front face: from each key position to the final position, weighted by recorded attention
      const mg = 1.5, cw = (b.w - 2 * mg) / N, chh = (b.h - 2 * mg) / N
      const yf = b.y + b.d + 0.6, zr = b.z + mg + chh * 0.5
      const xAt = (j: number) => b.x + mg + (j + 0.5) * cw
      for (let j = 0; j < N - 1; j++) {
        const s = row[j] / m
        if (s < 0.08) continue
        const ph = ((t * 0.8 + j * 0.19 + p.head! * 0.31) % 1)
        const x = xAt(j) + (xAt(N - 1) - xAt(j)) * ph
        const z = zr + Math.sin(ph * Math.PI) * chh * (1 + 2 * (N - 1 - j) / N)
        fb.glow(x, yf, z, scale(P.white, Math.min(1, s * boost)), s > 0.45 ? 1 : 0)
      }
      fb.glow(xAt(N - 1), yf, zr, scale(P.white, 0.4 * boost), 1)
    })
  }
  if (p.kind === 'pipe' && ctx.front !== null && act > 0) {
    q.push(() => {
      for (let i = 0; i < 3; i++) {
        const ph = (t * 2 + i / 3) % 1
        fb.glow(b.x + b.w * (b.w > 5 ? ph : 0.5), b.y + b.d * (b.d > 5 ? ph : 0.5), b.z + b.h * (b.h > 5 ? ph : 0.5), scale(P.lime, act), 1)
      }
    })
  }
  if (p.kind === 'post' && ctx.reveal.probs) {
    const pr = ctx.reveal.probs[p.index!] ?? 0
    if (pr > 0.04) q.push(() => fb.glow(b.x + b.w / 2, b.y + b.d / 2, b.z + b.h + 1, scale(probColor(pr), Math.min(1, pr * 1.6)), pr > 0.3 ? 2 : 1))
  }
  if (p.kind === 'cart' && act > 0.3) q.push(() => fb.glow(b.x + b.w / 2, b.y + b.d / 2, b.z + b.h + 1, scale(P.pink, act * 0.6), 1))
  if (ctx.front !== null && p.kind === 'ln' && act > 0.5) {
    q.push(() => {
      for (let i = 0; i < 10; i++) {
        const ph = i / 10 + t * 0.2
        const side = Math.floor(ph * 4) % 4, fr = (ph * 4) % 1
        const x = side === 0 ? b.x + b.w * fr : side === 1 ? b.x + b.w : side === 2 ? b.x + b.w * (1 - fr) : b.x
        const y = side === 0 ? b.y + b.d : side === 1 ? b.y + b.d * (1 - fr) : b.y
        fb.glow(x, y, b.z + b.h / 2, scale(P.cyan, act * 0.7), 0)
      }
    })
  }
}
