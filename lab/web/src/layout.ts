// Turns an architecture config into the physical machine: a list of boxes with stable keys,
// so the scene can animate parts between configurations (tween what persists, assemble what's
// new, eject what's gone).
//
// World units: the researcher is ~32 units tall. The smallest model is already a hulking
// van-sized machine; six layers of n_embd=128 tower ~500 units over the lab floor.

import type { Config } from './data'

export type Kind =
  | 'intake' | 'cart' | 'rail' | 'pos'
  | 'ln' | 'head' | 'proj' | 'ffn' | 'pipe' | 'adder'
  | 'lm' | 'post' | 'emitter'
  | 'outrail' | 'outcart'

export interface Box { x: number; y: number; z: number; w: number; d: number; h: number }

export interface Part {
  key: string
  kind: Kind
  comp: string // clickable component this part belongs to ('' = scenery)
  box: Box
  layer?: number
  head?: number
  index?: number
  spawnFrom?: string // key of a part to grow out of when this one appears
  delay?: number // assembly stagger (seconds)
}

export interface Layout {
  parts: Part[]
  W: number
  D: number
  hover: number
  top: number
  blockZ: number[] // base z of each block
  lanes: number
  T: number
}

export const HOVER = 40
export const BLOCK_H = 66
const SLOT = 7 // cartridge pitch along the conveyor
const LANE = 8 // conveyor lane pitch
const PER_LANE = 16
/** output cartridges on the exit belt (the most recent generated characters) */
export const OUT_SLOTS = 16

export function layoutFor(c: Config, prev?: Config): Layout {
  const parts: Part[] = []
  const W = Math.round(40 + 0.6 * c.n_embd)
  const D = Math.round(40 + 0.25 * c.n_embd)
  const x0 = -W / 2, y0 = -D / 2
  const z0 = HOVER
  const add = (p: Part) => parts.push(p)

  // ---- token embedding: intake hopper + cartridge conveyor ----
  add({ key: 'intake', kind: 'intake', comp: 'tok', box: { x: x0, y: y0, z: z0, w: W, d: D, h: 16 } })
  const lanes = Math.ceil(c.block_size / PER_LANE)
  const laneSpan = lanes * LANE
  const lx0 = -laneSpan / 2
  for (let l = 0; l < lanes; l++) {
    add({
      key: `rail${l}`, kind: 'rail', comp: 'tok', index: l, delay: l * 0.05,
      box: { x: lx0 + l * LANE + 1, y: D / 2, z: z0 + 3, w: LANE - 2, d: PER_LANE * SLOT + 6, h: 3 },
    })
  }
  for (let t = 0; t < c.block_size; t++) {
    const lane = Math.floor(t / PER_LANE), slot = t % PER_LANE
    add({
      key: `cart${t}`, kind: 'cart', comp: 'tok', index: t, delay: 0.1 + t * 0.008,
      box: { x: lx0 + lane * LANE + 1.5, y: D / 2 + 4 + (PER_LANE - 1 - slot) * SLOT, z: z0 + 6, w: LANE - 3, d: SLOT - 2, h: 4 },
    })
  }
  // anti-gravity emitters under the hopper
  for (const [i, [ex, ey]] of [[x0 + 6, y0 + 6], [x0 + W - 12, y0 + 6], [x0 + 6, y0 + D - 12], [x0 + W - 12, y0 + D - 12]].entries()) {
    add({ key: `emit${i}`, kind: 'emitter', comp: '', box: { x: ex, y: ey, z: z0 - 4, w: 6, d: 6, h: 4 } })
  }

  // ---- positional embedding band ----
  add({ key: 'pos', kind: 'pos', comp: 'pos', box: { x: x0 - 2, y: y0 - 2, z: z0 + 17, w: W + 4, d: D + 4, h: 9 } })

  // ---- transformer blocks ----
  const blockZ: number[] = []
  const prevHeads = prev?.n_head ?? c.n_head
  for (let l = 0; l < c.n_layer; l++) {
    const zb = z0 + 30 + l * BLOCK_H
    blockZ.push(zb)
    const dl = 0.08 * l
    const spawnAbove = l > 0 ? `b${l - 1}.ffn` : 'pos'
    add({ key: `b${l}.ln1`, kind: 'ln', comp: `b${l}.ln1`, layer: l, delay: dl, spawnFrom: spawnAbove, box: { x: x0 - 4, y: y0 - 4, z: zb, w: W + 8, d: D + 8, h: 3 } })

    const gap = 4
    const hw = (W - (c.n_head - 1) * gap) / c.n_head
    for (let h = 0; h < c.n_head; h++) {
      const src = Math.min(prevHeads - 1, Math.floor((h * prevHeads) / c.n_head))
      add({
        key: `b${l}.head${h}`, kind: 'head', comp: `b${l}.head${h}`, layer: l, head: h, delay: dl + 0.05 + h * 0.06,
        spawnFrom: `b${l}.head${src}`,
        box: { x: x0 + h * (hw + gap), y: y0 + 4, z: zb + 5, w: hw, d: D - 8, h: 22 },
      })
    }
    add({ key: `b${l}.proj`, kind: 'proj', comp: `b${l}.proj`, layer: l, delay: dl + 0.1, spawnFrom: spawnAbove, box: { x: x0 + 2, y: y0 + 2, z: zb + 27, w: W - 4, d: D - 4, h: 4 } })
    add({ key: `b${l}.ln2`, kind: 'ln', comp: `b${l}.ln2`, layer: l, delay: dl + 0.12, spawnFrom: spawnAbove, box: { x: x0 - 4, y: y0 - 4, z: zb + 33, w: W + 8, d: D + 8, h: 3 } })
    add({ key: `b${l}.ffn`, kind: 'ffn', comp: `b${l}.ffn`, layer: l, delay: dl + 0.15, spawnFrom: spawnAbove, box: { x: x0 - 5, y: y0 - 3, z: zb + 38, w: W + 10, d: D + 6, h: 26 } })

    // residual 1 bypasses ln1 + attention on the +x face
    const r1x = x0 + W + 8, r1y = y0 + D * 0.3
    const r1z0 = zb - 2, r1z1 = zb + 32
    const res1 = `b${l}.res1`
    add({ key: `${res1}.out`, kind: 'pipe', comp: res1, layer: l, delay: dl + 0.2, box: { x: x0 + W - 2, y: r1y, z: r1z0, w: 14, d: 4, h: 4 } })
    add({ key: `${res1}.run`, kind: 'pipe', comp: res1, layer: l, delay: dl + 0.22, index: 1, box: { x: r1x, y: r1y, z: r1z0, w: 4, d: 4, h: r1z1 - r1z0 + 4 } })
    add({ key: `${res1}.in`, kind: 'adder', comp: res1, layer: l, delay: dl + 0.24, box: { x: x0 + W - 2, y: r1y - 1, z: r1z1 - 1, w: 14, d: 6, h: 6 } })

    // residual 2 bypasses ln2 + feedforward on the +y face (outside the bulkier ffn)
    const r2y = y0 + D + 8, r2x = x0 + W * 0.25
    const r2z0 = zb + 31, r2z1 = zb + BLOCK_H - 1
    const res2 = `b${l}.res2`
    add({ key: `${res2}.out`, kind: 'pipe', comp: res2, layer: l, delay: dl + 0.26, box: { x: r2x, y: y0 + D - 2, z: r2z0, w: 4, d: 14, h: 4 } })
    add({ key: `${res2}.run`, kind: 'pipe', comp: res2, layer: l, delay: dl + 0.28, index: 1, box: { x: r2x, y: r2y, z: r2z0, w: 4, d: 4, h: r2z1 - r2z0 + 4 } })
    add({ key: `${res2}.in`, kind: 'adder', comp: res2, layer: l, delay: dl + 0.3, box: { x: r2x - 1, y: y0 + D - 2, z: r2z1 - 1, w: 6, d: 14, h: 6 } })
  }

  // ---- final layer norm + output projection ----
  const zt = z0 + 30 + c.n_layer * BLOCK_H + 2
  add({ key: 'lnf', kind: 'ln', comp: 'lnf', box: { x: x0 - 5, y: y0 - 5, z: zt, w: W + 10, d: D + 10, h: 3 } })
  add({ key: 'lm', kind: 'lm', comp: 'lm', box: { x: x0, y: y0, z: zt + 5, w: W, d: D, h: 12 } })
  const COLS = 13, ROWS = 5
  const pw = (W - 8) / COLS, pd = (D - 8) / ROWS
  for (let i = 0; i < COLS * ROWS; i++) {
    const cx = i % COLS, cy = Math.floor(i / COLS)
    add({
      key: `post${i}`, kind: 'post', comp: 'lm', index: i, delay: 0.2 + i * 0.006,
      box: { x: x0 + 4 + cx * pw + pw * 0.2, y: y0 + 4 + cy * pd + pd * 0.2, z: zt + 17, w: pw * 0.6, d: pd * 0.6, h: 4 },
    })
  }

  // ---- output: an exit belt leaves the opposite (+x) side of the output projection ----
  const oy = y0 + D / 2 - 4
  add({ key: 'outrail', kind: 'outrail', comp: 'out', delay: 0.3, box: { x: x0 + W - 1, y: oy - 0.5, z: zt + 8, w: OUT_SLOTS * SLOT + 8, d: 9, h: 3 } })
  for (let s = 0; s < OUT_SLOTS; s++) {
    add({
      key: `out${s}`, kind: 'outcart', comp: 'out', index: s, delay: 0.35 + s * 0.012,
      box: { x: x0 + W + 4 + s * SLOT, y: oy + 1.5, z: zt + 11, w: SLOT - 2, d: 5, h: 4 },
    })
  }

  return { parts, W, D, hover: z0, top: zt + 21, blockZ, lanes, T: c.block_size }
}
