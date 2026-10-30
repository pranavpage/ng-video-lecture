// Endesga 32 — every surface colour in the lab is drawn from this list.
// Colours are stored as 0xAABBGGRR so they can be written straight into an ImageData view.

export function hex(h: string): number {
  const n = parseInt(h.replace('#', ''), 16)
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255
  return (0xff000000 | (b << 16) | (g << 8) | r) >>> 0
}

export function rgbOf(c: number): [number, number, number] {
  return [c & 255, (c >> 8) & 255, (c >> 16) & 255]
}

export function pack(r: number, g: number, b: number): number {
  return (0xff000000 | ((b & 255) << 16) | ((g & 255) << 8) | (r & 255)) >>> 0
}

export function mix(a: number, b: number, t: number): number {
  const [ar, ag, ab] = rgbOf(a), [br, bg, bb] = rgbOf(b)
  return pack(ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t)
}

export const P = {
  ink: hex('181425'),
  night: hex('262b44'),
  slate: hex('3a4466'),
  steel: hex('5a6988'),
  silver: hex('8b9bb4'),
  mist: hex('c0cbdc'),
  white: hex('ffffff'),

  deepSea: hex('124e89'),
  sea: hex('0099db'),
  cyan: hex('2ce8f5'),

  plum: hex('3e2731'),
  rust: hex('733e39'),
  brick: hex('be4a2f'),
  copper: hex('d77643'),
  tan: hex('e4a672'),
  sand: hex('ead4aa'),
  clay: hex('b86f50'),

  wine: hex('a22633'),
  red: hex('e43b44'),
  hotRed: hex('ff0044'),
  orange: hex('f77622'),
  amber: hex('feae34'),
  lemon: hex('fee761'),

  moss: hex('193c3e'),
  pine: hex('265c42'),
  green: hex('3e8948'),
  lime: hex('63c74d'),

  grape: hex('68386c'),
  magenta: hex('b55088'),
  pink: hex('f6757a'),
  peach: hex('e8b796'),
  bronze: hex('c28569'),
}

// light-to-dark ramps used for the three visible faces of a box (top, right, left)
export type Ramp = [number, number, number, number] // highlight, top, right, left
export const RAMPS = {
  metal: [P.mist, P.silver, P.steel, P.slate] as Ramp,
  darkMetal: [P.steel, P.slate, P.night, P.ink] as Ramp,
  crystal: [P.white, P.cyan, P.sea, P.deepSea] as Ramp,
  rust: [P.tan, P.copper, P.brick, P.rust] as Ramp,
  conduit: [P.lime, P.green, P.pine, P.moss] as Ramp,
  cartridge: [P.peach, P.pink, P.magenta, P.grape] as Ramp,
  brass: [P.lemon, P.amber, P.orange, P.clay] as Ramp,
  floor: [P.slate, P.night, P.night, P.ink] as Ramp,
  wall: [P.steel, P.slate, P.night, P.ink] as Ramp,
  terminal: [P.sand, P.tan, P.clay, P.rust] as Ramp,
}
