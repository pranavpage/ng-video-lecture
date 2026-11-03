// The laboratory: floor, walls, racks, cabling, the control terminal and its researcher.

import { FB } from './fb'
import { P, RAMPS, mix } from './palette'
import { box, textMask, hash, nextGroup, FaceShader } from './draw3d'
import type { Box } from './layout'
import type { Picker } from './pick'

export const ROOM = 260
const WALL_H = 190
const PAD = 82

// the terminal desk sits in the foreground, front-right of the machine
export const DESK: Box = { x: 70, y: 56, z: 0, w: 34, d: 16, h: 14 }
export const CRT: Box = { x: 78, y: 58, z: 14, w: 14, d: 12, h: 13 }
export const RESEARCHER = { x: 112, y: 68 }
/** world-space bounds of the foreground props, so the camera keeps them in frame */
export const FOREGROUND: Box = { x: 68, y: 52, z: 0, w: 52, d: 30, h: 34 }

const sign = textMask('TINYGPT LAB', 5)
const subSign = textMask('TRANSFORMER RESEARCH BAY 03', 2)
const wallSign = textMask('CAUTION: LEVITATING MODEL', 2)

const floorShader: FaceShader = (_f, a, b) => {
  const x = a - ROOM, y = b - ROOM
  // hazard border around the landing pad under the machine
  const ax = Math.abs(x), ay = Math.abs(y), m = Math.max(ax, ay)
  if (m > PAD && m < PAD + 6) return ((Math.floor((x + y) / 4) & 1) ? P.amber : P.ink)
  if (m <= PAD) {
    // pad: dark plating with concentric guide rings
    const r = Math.hypot(x, y)
    if (Math.abs(r - 30) < 0.8 || Math.abs(r - 58) < 0.8) return P.deepSea
    if (ax < 0.6 || ay < 0.6) return P.slate
    return ((Math.floor(x / 8) + Math.floor(y / 8)) & 1) ? P.night : mix(P.night, P.ink, 0.35)
  }
  const tx = ((a % 16) + 16) % 16, ty = ((b % 16) + 16) % 16
  if (tx < 0.8 || ty < 0.8) return P.ink
  const tile = (Math.floor(a / 16) * 31 + Math.floor(b / 16) * 17) | 0
  const n = hash(tile)
  return n > 0.85 ? mix(P.slate, P.night, 0.5) : n > 0.15 ? P.night : mix(P.night, P.ink, 0.4)
}

function wallShader(signs: boolean): FaceShader {
  return (_f, a, b) => {
    if (b < 8) return ((Math.floor((a + b) / 5) & 1) ? P.amber : P.ink)
    if (b < 9) return P.ink
    if (b > WALL_H - 3) return P.steel
    // horizontal coolant pipes
    if (b > 128 && b < 134) return b > 132 ? P.lime : b > 130 ? P.green : P.pine
    if (b > 138 && b < 142) return b > 141 ? P.sea : P.deepSea
    if (signs) {
      const sx = a - 150, sy = b - 66
      if (sx > -6 && sx < sign.width + 6 && sy > -6 && sy < sign.height + 6) {
        if (sign.at(sx, sy)) return (Math.floor(sy / 5) & 1) ? P.amber : P.lemon
        if (sx < -4 || sy < -4 || sx > sign.width + 4 || sy > sign.height + 4) return P.clay
        return P.ink
      }
      if (subSign.at(a - 150, b - 52)) return P.silver
    } else {
      if (wallSign.at(a - 180, b - 100)) return P.amber
    }
    // riveted panels
    const pa = ((a % 40) + 40) % 40
    if (pa < 1) return P.ink
    if (pa < 2) return P.steel
    if ((pa > 3 && pa < 4.2 || pa > 36 && pa < 37.2) && (b % 24 > 3 && b % 24 < 5)) return P.silver
    const band = Math.floor(b / 48)
    return band & 1 ? P.slate : mix(P.slate, P.night, 0.35)
  }
}

const solid = (ramp: readonly number[]): FaceShader => (f, a, b, fw, fh) => {
  const top = f === 0
  if (top && (a < 0.7 || b < 0.7 || a > fw - 0.7 || b > fh - 0.7)) return ramp[0]
  if (!top && b > fh - 0.7) return ramp[0]
  return ramp[top ? 1 : f === 1 ? 2 : 3]
}

function rackShader(seed: number, t: number): FaceShader {
  return (f, a, b, fw, fh) => {
    if (f === 0) return a < 0.8 || b < 0.8 ? P.steel : P.night
    if (b > fh - 1) return P.steel
    // a server rack face: drive slots and blinking status LEDs
    const slot = Math.floor(b / 5)
    const yb = b % 5
    if (yb < 0.8) return P.ink
    if (a < 1.5 || a > fw - 1.5) return P.slate
    if (yb > 2 && yb < 3.2 && a > 3 && a < 4.5) {
      const on = hash(seed * 97 + slot * 13 + Math.floor(t * (1 + hash(slot + seed) * 3))) > 0.45
      return on ? (hash(slot + seed * 7) > 0.7 ? P.amber : P.lime) : P.pine
    }
    return f === 1 ? P.night : mix(P.night, P.ink, 0.4)
  }
}

/** floor and walls: static, so main caches them while the camera is still */
export function drawRoomShell(fb: FB) {
  const R = ROOM
  fb.face([-R, -R, 0], [2 * R, 0, 0], [0, 2 * R, 0], (a, b) => floorShader(0, a, b, 0, 0, 0, 0), 0, 1)
  fb.face([-R, -R, 0], [0, 2 * R, 0], [0, 0, WALL_H], (a, b) => wallShader(false)(1, 2 * R - a, b, 0, 0, 0, 0), 0, 2)
  fb.face([-R, -R, 0], [2 * R, 0, 0], [0, 0, WALL_H], (a, b) => wallShader(true)(2, a, b, 0, 0, 0, 0), 0, 3)
}

export function drawRoom(fb: FB, t: number, picker: Picker, termActive: number) {
  const R = ROOM

  // server racks along the back walls
  for (let i = 0; i < 5; i++) {
    box(fb, { x: -R + 1, y: -R + 30 + i * 26, z: 0, w: 18, d: 22, h: 70 + (i % 2) * 14 }, rackShader(i, t), 0, nextGroup())
  }
  for (let i = 0; i < 3; i++) {
    box(fb, { x: -R + 300 + i * 28, y: -R + 1, z: 0, w: 24, d: 16, h: 58 }, rackShader(10 + i, t), 0, nextGroup())
  }
  // crates and a coolant tank
  box(fb, { x: -R + 40, y: -R + 200, z: 0, w: 20, d: 20, h: 16 }, solid(RAMPS.terminal), 0, nextGroup())
  box(fb, { x: -R + 44, y: -R + 204, z: 16, w: 12, d: 12, h: 10 }, solid(RAMPS.terminal), 0, nextGroup())
  box(fb, { x: -R + 200, y: -R + 30, z: 0, w: 26, d: 26, h: 46 }, solid(RAMPS.conduit), 0, nextGroup())
  box(fb, { x: -R + 203, y: -R + 33, z: 46, w: 20, d: 20, h: 4 }, solid(RAMPS.metal), 0, nextGroup())

  // floor cabling from the terminal to the landing pad
  const cable = solid([P.slate, P.night, P.ink, P.ink])
  box(fb, { x: 48, y: 66, z: 0, w: 22, d: 2, h: 1.2 }, cable, 0, nextGroup())
  box(fb, { x: 48, y: 30, z: 0, w: 2, d: 38, h: 1.2 }, cable, 0, nextGroup())
  box(fb, { x: 56, y: 72, z: 0, w: 14, d: 2, h: 1.2 }, cable, 0, nextGroup())
  box(fb, { x: 56, y: 40, z: 0, w: 2, d: 34, h: 1.2 }, cable, 0, nextGroup())

  // desk + CRT terminal (clickable: it is the control interface)
  const tp = picker.world('terminal')
  const desk = solid(RAMPS.terminal)
  box(fb, { x: DESK.x, y: DESK.y, z: DESK.h - 2, w: DESK.w, d: DESK.d, h: 2 }, desk, tp, nextGroup())
  for (const [lx, ly] of [[0, 0], [DESK.w - 2, 0], [0, DESK.d - 2], [DESK.w - 2, DESK.d - 2]])
    box(fb, { x: DESK.x + lx, y: DESK.y + ly, z: 0, w: 2, d: 2, h: DESK.h - 2 }, solid(RAMPS.darkMetal), tp, nextGroup())
  box(fb, { x: DESK.x + 2, y: DESK.y + 2, z: 4, w: 10, d: DESK.d - 4, h: 7 }, solid(RAMPS.darkMetal), tp, nextGroup())
  const g = nextGroup()
  box(fb, CRT, (f, a, b, fw, fh) => {
    if (f === 2 && a > 1.5 && a < fw - 1.5 && b > 2 && b < fh - 1.5) {
      // phosphor screen: scrolling lines of text
      const row = Math.floor((b - 2) / 1.6)
      if ((b - 2) % 1.6 < 0.6) return mix(P.moss, P.ink, 0.5)
      const len = hash(row + Math.floor(t * 2)) * (fw - 4)
      const lit = a - 2 < len
      return lit ? (termActive > 0 ? P.lime : P.green) : P.moss
    }
    return solid(RAMPS.terminal)(f, a, b, fw, fh, 0, 0)
  }, tp, g)
  box(fb, { x: CRT.x + 2, y: CRT.y - 4, z: CRT.z + 1, w: 10, d: 4, h: 10 }, solid(RAMPS.terminal), tp, nextGroup())
  box(fb, { x: DESK.x + 4, y: DESK.y + DESK.d - 3, z: DESK.h, w: 16, d: 2.5, h: 1 }, (f, a) =>
    f === 0 ? ((Math.floor(a) & 1) ? P.mist : P.silver) : P.steel, tp, nextGroup())
  box(fb, { x: DESK.x + 23, y: DESK.y + 5, z: DESK.h, w: 6, d: 5, h: 3 }, solid([P.white, P.mist, P.silver, P.steel]), tp, nextGroup())

  drawResearcher(fb, RESEARCHER.x, RESEARCHER.y, t, tp)
}

function drawResearcher(fb: FB, x: number, y: number, t: number, pick: number) {
  // seen from behind, facing the machine (toward -x, -y)
  const bob = Math.sin(t * 2) > 0.95 ? 0.6 : 0
  const trousers = solid([P.slate, P.night, P.night, P.ink])
  const coat = solid([P.white, P.mist, P.silver, P.steel])
  const skin = solid([P.peach, P.peach, P.bronze, P.clay])
  const hair = solid([P.clay, P.rust, P.rust, P.plum])
  const shoe = solid([P.night, P.ink, P.ink, P.ink])
  box(fb, { x: x, y: y, z: 0, w: 3.5, d: 3.5, h: 1.5 }, shoe, pick, nextGroup())
  box(fb, { x: x, y: y + 4.5, z: 0, w: 3.5, d: 3.5, h: 1.5 }, shoe, pick, nextGroup())
  box(fb, { x: x + 0.3, y: y + 0.3, z: 1.5, w: 3, d: 3, h: 11 }, trousers, pick, nextGroup())
  box(fb, { x: x + 0.3, y: y + 4.8, z: 1.5, w: 3, d: 3, h: 11 }, trousers, pick, nextGroup())
  box(fb, { x: x - 1, y: y - 0.5, z: 9, w: 6, d: 9, h: 5 }, coat, pick, nextGroup())
  box(fb, { x: x - 0.5, y: y, z: 14, w: 5, d: 8, h: 9 + bob }, coat, pick, nextGroup())
  box(fb, { x: x, y: y - 2.5, z: 13, w: 2.5, d: 2.5, h: 9 }, coat, pick, nextGroup())
  box(fb, { x: x, y: y + 8, z: 13, w: 2.5, d: 2.5, h: 9 }, coat, pick, nextGroup())
  box(fb, { x: x + 0.2, y: y - 2.3, z: 11.5, w: 2, d: 2, h: 1.5 }, skin, pick, nextGroup())
  box(fb, { x: x + 0.2, y: y + 8.2, z: 11.5, w: 2, d: 2, h: 1.5 }, skin, pick, nextGroup())
  box(fb, { x: x + 0.5, y: y + 2.5, z: 23 + bob, w: 3, d: 3, h: 1.5 }, skin, pick, nextGroup())
  box(fb, { x: x - 0.5, y: y + 1, z: 24.5 + bob, w: 5.5, d: 6, h: 5.5 }, (f, a, b, fw, fh) => {
    if (f === 0) return P.rust
    if (b > fh - 1.8) return f === 1 ? P.rust : P.clay
    // back of the head is hair; a sliver of face on the +y side
    if (f === 2 && a < 1.6) return P.bronze
    return f === 1 ? P.rust : P.clay
  }, pick, nextGroup())
  box(fb, { x: x - 0.8, y: y + 0.7, z: 29.6 + bob, w: 6, d: 6.6, h: 1.4 }, hair, pick, nextGroup())
}
