import { FB, Mode } from './fb'
import type { Box } from './layout'
import { FONT } from './font'

/** face 0 = top, 1 = +x side (screen right), 2 = +y side (screen left) */
export type FaceShader = (f: 0 | 1 | 2, a: number, b: number, fw: number, fh: number, px: number, py: number) => number

const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
export function bayer(px: number, py: number): number {
  return BAYER[(px & 3) + ((py & 3) << 2)] / 16
}

let groupCounter = 10
export function resetGroups() { groupCounter = 10 }
export function nextGroup(): number { groupCounter = groupCounter >= 65000 ? 10 : groupCounter + 1; return groupCounter }

/** draws the three camera-facing faces of an axis-aligned box. `vis` < 1 dissolves it with ordered dither */
export function box(fb: FB, b: Box, sh: FaceShader, pick: number, group: number, vis = 1, mode: Mode = Mode.Opaque) {
  if (b.w <= 0.05 || b.d <= 0.05 || b.h <= 0.05 || vis <= 0) return
  const dissolve = vis < 1
  const wrap = (f: 0 | 1 | 2, fw: number, fh: number) => (a: number, bb: number, px: number, py: number) => {
    if (dissolve && bayer(px, py) >= vis) return 0
    return sh(f, a, bb, fw, fh, px, py)
  }
  fb.face([b.x, b.y, b.z + b.h], [b.w, 0, 0], [0, b.d, 0], wrap(0, b.w, b.d), pick, group, mode)
  fb.face([b.x + b.w, b.y, b.z], [0, b.d, 0], [0, 0, b.h], wrap(1, b.d, b.h), pick, group, mode)
  fb.face([b.x, b.y + b.d, b.z], [b.w, 0, 0], [0, 0, b.h], wrap(2, b.w, b.h), pick, group, mode)
}

/** bitmap text painted onto a face: returns true where a glyph pixel lies under (a, b) */
export function textMask(s: string, cell: number) {
  const W = s.length * 6 - 1
  const bits = new Uint8Array(W * 7)
  for (let i = 0; i < s.length; i++) {
    for (const [gx, gy] of FONT.get(s[i]) ?? []) if (gy < 7) bits[gy * W + i * 6 + gx] = 1
  }
  return {
    width: W * cell,
    height: 7 * cell,
    at(a: number, b: number): boolean {
      // b grows upward on side faces, glyph rows grow downward
      const gx = Math.floor(a / cell), gy = 6 - Math.floor(b / cell)
      if (gx < 0 || gy < 0 || gx >= W || gy > 6) return false
      return bits[gy * W + gx] === 1
    },
  }
}

export function clamp01(x: number) { return x < 0 ? 0 : x > 1 ? 1 : x }
export function smooth(x: number) { x = clamp01(x); return x * x * (3 - 2 * x) }
export function hash(n: number): number {
  n = (n ^ 61) ^ (n >>> 16)
  n = n + (n << 3)
  n = n ^ (n >>> 4)
  n = Math.imul(n, 0x27d4eb2d)
  n = n ^ (n >>> 15)
  return (n >>> 0) / 4294967295
}
