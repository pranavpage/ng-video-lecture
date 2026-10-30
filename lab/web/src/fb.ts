// A tiny software rasterizer for the isometric lab.
//
// World space is z-up. The 2:1 isometric projection is
//   sx = cx + (x - y) * zoom
//   sy = cy + ((x + y) / 2 - z) * zoom
// so the view ray is (1, 1, 1) and `x + y + z` is a perfect depth value (larger = nearer).
// Every face is a parallelogram rasterized with a z-buffer, a pick buffer (component id per
// pixel, used for clicking) and a group buffer (used for the pixel-art silhouette outline pass).

import { FONT, MICRO_FONT, CHAR_W, glyph } from './font'
import { mix, P, rgbOf, pack } from './palette'

export type V3 = [number, number, number]
/** returns a colour, or 0 for "leave this pixel alone" */
export type Shader = (a: number, b: number, px: number, py: number) => number

export const enum Mode { Opaque, Add, Shadow }

export class FB {
  w = 0
  h = 0
  color = new Uint32Array(0)
  depth = new Float32Array(0)
  pick = new Uint16Array(0)
  group = new Uint16Array(0)
  img!: ImageData
  cx = 0
  cy = 0
  zoom = 1

  resize(w: number, h: number) {
    if (w === this.w && h === this.h) return
    this.w = w
    this.h = h
    this.img = new ImageData(w, h)
    this.color = new Uint32Array(this.img.data.buffer)
    this.depth = new Float32Array(w * h)
    this.pick = new Uint16Array(w * h)
    this.group = new Uint16Array(w * h)
  }

  private snap: { key: string; color: Uint32Array; depth: Float32Array; pick: Uint16Array; group: Uint16Array } | null = null

  /** restore a cached static layer for this camera; returns false if it must be redrawn */
  restore(key: string): boolean {
    const s = this.snap
    if (!s || s.key !== key || s.color.length !== this.color.length) return false
    this.color.set(s.color); this.depth.set(s.depth); this.pick.set(s.pick); this.group.set(s.group)
    return true
  }

  snapshot(key: string) {
    this.snap = { key, color: this.color.slice(), depth: this.depth.slice(), pick: this.pick.slice(), group: this.group.slice() }
  }

  clear(c: number) {
    this.color.fill(c)
    this.depth.fill(-Infinity)
    this.pick.fill(0)
    this.group.fill(0)
  }

  sx(x: number, y: number): number { return this.cx + (x - y) * this.zoom }
  sy(x: number, y: number, z: number): number { return this.cy + ((x + y) * 0.5 - z) * this.zoom }

  /**
   * Rasterize the parallelogram p0, p0+u, p0+u+v, p0+v.
   * The shader receives (a, b): world-unit distances along u and v.
   */
  face(p0: V3, u: V3, v: V3, shade: Shader, pick = 0, group = 0, mode: Mode = Mode.Opaque) {
    const z = this.zoom
    const sx0 = this.cx + (p0[0] - p0[1]) * z
    const sy0 = this.cy + ((p0[0] + p0[1]) * 0.5 - p0[2]) * z
    const sux = (u[0] - u[1]) * z, suy = ((u[0] + u[1]) * 0.5 - u[2]) * z
    const svx = (v[0] - v[1]) * z, svy = ((v[0] + v[1]) * 0.5 - v[2]) * z
    const det = sux * svy - suy * svx
    if (Math.abs(det) < 1e-4) return
    const lenU = Math.hypot(u[0], u[1], u[2]), lenV = Math.hypot(v[0], v[1], v[2])
    const d0 = p0[0] + p0[1] + p0[2], du = u[0] + u[1] + u[2], dv = v[0] + v[1] + v[2]

    const ys = [sy0, sy0 + suy, sy0 + suy + svy, sy0 + svy]
    const y0 = Math.max(0, Math.floor(Math.min(...ys)))
    const y1 = Math.min(this.h - 1, Math.ceil(Math.max(...ys)))
    const xs = [sx0, sx0 + sux, sx0 + sux + svx, sx0 + svx]
    const xmin = Math.max(0, Math.floor(Math.min(...xs)))
    const xmax = Math.min(this.w - 1, Math.ceil(Math.max(...xs)))
    if (xmin > xmax || y0 > y1) return

    const A1 = svy / det, B1 = -suy / det
    const W = this.w, color = this.color, depth = this.depth, pk = this.pick, gp = this.group
    for (let py = y0; py <= y1; py++) {
      const fy = py + 0.5 - sy0
      const A0 = -svx * fy / det, B0 = sux * fy / det
      // a = A1*fx + A0 in [0,1), b = B1*fx + B0 in [0,1)  ->  interval in fx
      let lo = -Infinity, hi = Infinity
      if (A1 > 1e-9) { lo = Math.max(lo, -A0 / A1); hi = Math.min(hi, (1 - A0) / A1) }
      else if (A1 < -1e-9) { lo = Math.max(lo, (1 - A0) / A1); hi = Math.min(hi, -A0 / A1) }
      else if (A0 < 0 || A0 >= 1) continue
      if (B1 > 1e-9) { lo = Math.max(lo, -B0 / B1); hi = Math.min(hi, (1 - B0) / B1) }
      else if (B1 < -1e-9) { lo = Math.max(lo, (1 - B0) / B1); hi = Math.min(hi, -B0 / B1) }
      else if (B0 < 0 || B0 >= 1) continue
      const pxa = Math.max(xmin, Math.ceil(lo + sx0 - 0.5))
      const pxb = Math.min(xmax, Math.ceil(hi + sx0 - 0.5) - 1)
      let idx = py * W + pxa
      for (let px = pxa; px <= pxb; px++, idx++) {
        const fx = px + 0.5 - sx0
        let a = A1 * fx + A0, b = B1 * fx + B0
        a = a < 0 ? 0 : a > 0.99999 ? 0.99999 : a
        b = b < 0 ? 0 : b > 0.99999 ? 0.99999 : b
        const d = d0 + a * du + b * dv
        if (d < depth[idx]) continue
        const c = shade(a * lenU, b * lenV, px, py)
        if (c === 0) continue
        if (mode === Mode.Shadow) {
          color[idx] = mix(color[idx], P.ink, ((px + py) & 1) ? 0.3 : 0.6)
          continue
        }
        if (mode === Mode.Add) {
          color[idx] = add(color[idx], c)
          continue
        }
        color[idx] = c
        depth[idx] = d
        pk[idx] = pick
        gp[idx] = group
      }
    }
  }

  /** world-space glowing point (additive square sprite), depth-tested but not depth-writing */
  glow(x: number, y: number, z: number, c: number, r = 1) {
    const sx = Math.round(this.sx(x, y)), sy = Math.round(this.sy(x, y, z))
    const d = x + y + z
    const rr = Math.max(0, Math.round(r))
    for (let dy = -rr; dy <= rr; dy++) for (let dx = -rr; dx <= rr; dx++) {
      if (Math.abs(dx) + Math.abs(dy) > rr) continue
      const px = sx + dx, py = sy + dy
      if (px < 0 || py < 0 || px >= this.w || py >= this.h) continue
      const i = py * this.w + px
      if (d + 2 < this.depth[i]) continue
      this.color[i] = add(this.color[i], c)
    }
  }

  /** silhouette outlines: darken pixels whose neighbour belongs to another group further away */
  outline(hot: Uint8Array, hiColor: number) {
    const W = this.w, H = this.h, g = this.group, d = this.depth, c = this.color, pk = this.pick
    if (this.scratch.length !== c.length) this.scratch = new Uint32Array(c.length)
    const out = this.scratch
    out.set(c)
    const nb = [-1, 1, -W, W]
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const i = y * W + x
        const gi = g[i]
        if (gi === 0) continue
        const di = d[i] - 1.5
        const isHot = hot[pk[i]] === 1
        let edge = false, hotEdge = false
        for (let k = 0; k < 4; k++) {
          const j = i + nb[k]
          if (isHot && hot[pk[j]] !== 1) hotEdge = true
          if (g[j] !== gi && d[j] < di) edge = true
        }
        if (hotEdge) out[i] = hiColor
        else if (edge) out[i] = mix(c[i], P.ink, 0.75)
      }
    }
    c.set(out)
  }
  scratch = new Uint32Array(0)

  // ---------- screen-space 2D primitives for the UI ----------

  rect(x: number, y: number, w: number, h: number, col: number, pick = 0) {
    x = Math.round(x); y = Math.round(y)
    const x0 = Math.max(0, x), y0 = Math.max(0, y)
    const x1 = Math.min(this.w, x + w), y1 = Math.min(this.h, y + h)
    for (let py = y0; py < y1; py++) {
      const row = py * this.w
      for (let px = x0; px < x1; px++) {
        this.color[row + px] = col
        if (pick) this.pick[row + px] = pick
        else this.pick[row + px] = 0
      }
    }
  }

  /** translucent-looking fill via ordered dither */
  dither(x: number, y: number, w: number, h: number, col: number, pick = 0) {
    x = Math.round(x); y = Math.round(y)
    for (let py = Math.max(0, y); py < Math.min(this.h, y + h); py++)
      for (let px = Math.max(0, x); px < Math.min(this.w, x + w); px++) {
        const i = py * this.w + px
        if (((px + py) & 1) === 0) this.color[i] = col
        else this.color[i] = mix(this.color[i], col, 0.5)
        this.pick[i] = pick
      }
  }

  frame(x: number, y: number, w: number, h: number, col: number) {
    this.rect(x, y, w, 1, col)
    this.rect(x, y + h - 1, w, 1, col)
    this.rect(x, y, 1, h, col)
    this.rect(x + w - 1, y, 1, h, col)
  }

  px(x: number, y: number, col: number) {
    x = Math.round(x); y = Math.round(y)
    if (x < 0 || y < 0 || x >= this.w || y >= this.h) return
    this.color[y * this.w + x] = col
  }

  line(x0: number, y0: number, x1: number, y1: number, col: number) {
    x0 = Math.round(x0); y0 = Math.round(y0); x1 = Math.round(x1); y1 = Math.round(y1)
    const dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0)
    const sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1
    let err = dx + dy
    for (let n = 0; n < 4000; n++) {
      this.px(x0, y0, col)
      if (x0 === x1 && y0 === y1) break
      const e2 = 2 * err
      if (e2 >= dy) { err += dy; x0 += sx }
      if (e2 <= dx) { err += dx; y0 += sy }
    }
  }

  text(x: number, y: number, s: string, col: number, scale = 1, shadow = 0) {
    x = Math.round(x); y = Math.round(y)
    if (shadow) this.text(x + scale, y + scale, s, shadow, scale)
    for (let i = 0; i < s.length; i++) {
      const g = glyph(s[i])
      const ox = x + i * CHAR_W * scale
      if (ox > this.w) break
      for (const [gx, gy] of g) {
        if (scale === 1) this.px(ox + gx, y + gy, col)
        else for (let sy = 0; sy < scale; sy++) for (let sx = 0; sx < scale; sx++)
          this.px(ox + gx * scale + sx, y + gy * scale + sy, col)
      }
    }
  }

  micro(x: number, y: number, s: string, col: number) {
    x = Math.round(x); y = Math.round(y)
    for (let i = 0; i < s.length; i++) {
      const g = MICRO_FONT.get(s[i])
      if (!g) continue
      for (const [gx, gy] of g) this.px(x + i * 4 + gx, y + gy, col)
    }
  }
}

export function add(a: number, b: number): number {
  const [ar, ag, ab] = rgbOf(a), [br, bg, bb] = rgbOf(b)
  return pack(Math.min(255, ar + br), Math.min(255, ag + bg), Math.min(255, ab + bb))
}

export function scale(c: number, k: number): number {
  const [r, g, b] = rgbOf(c)
  return pack(r * k, g * k, b * k)
}

export { FONT }
