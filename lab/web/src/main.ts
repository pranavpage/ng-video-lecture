import { FB } from './fb'
import { P } from './palette'
import { Picker } from './pick'
import { resetGroups } from './draw3d'
import { drawRoom, drawRoomShell, FOREGROUND, CRT } from './room'
import { Scene } from './scene'
import { layoutFor, Box } from './layout'
import { Config, Param, PARAMS, Run, RunsFile, loadRuns, loadDetail, runId, dropoutOptions } from './data'
import { describe } from './info'
import { drawControl, drawInfo, drawTooltip, drawHint, drawPrintout, drawNotebook, CONTROL, NOTEBOOK_W } from './ui'
import { Experiment, advance, machineState, phaseOf } from './experiment'
import { drawScale } from './scale'

const canvas = document.getElementById('screen') as HTMLCanvasElement
const ctx2d = canvas.getContext('2d')!
const fb = new FB()
const picker = new Picker()
const scene = new Scene()
const hot = new Uint8Array(65536)

const FALLBACK_GRID: Record<Param, number[]> = { n_layer: [1, 2, 4, 6], n_head: [1, 2, 4], n_embd: [32, 64, 128], block_size: [16, 32, 64, 128] }

const S = {
  data: null as RunsFile | null,
  loadError: '',
  cfg: { n_layer: 2, n_head: 2, n_embd: 64, block_size: 32 } as Config,
  row: 0,
  selected: null as string | null,
  hovered: null as string | null,
  hoverSub: -1,
  hoverUi: false,
  hoverHint: '',
  mx: -1,
  my: -1,
  /** the experiment currently on the printout (and, if its config is dialled in, on the machine) */
  exp: null as Experiment | null,
  /** every run replayed this session, for the notebook (in memory only) */
  history: [] as Run[],
  printoutOpen: true,
  notebookOpen: true,
  /** play generation at the measured per-token latency instead of slow motion */
  realtime: false,
  loading: false,
  /** the scale chart (shares the top-left slot with component cards) */
  scaleOpen: false,
  cam: { sx: 0, sy: 0, zoom: 1, init: false },
  pixel: 3,
}

const grid = () => S.data?.meta.grid ?? FALLBACK_GRID
const total = () => {
  // the full grid, plus the dropout variants of the study corner
  const g = grid(), st = S.data?.meta.dropout_study
  const all = PARAMS.reduce((a, p) => a * g[p].length, 1)
  if (!st) return all
  const corner = PARAMS.reduce((a, p) => a * (p in st.where ? 1 : g[p].length), 1)
  return all + corner * st.dropout.length
}
const runFor = (c: Config) => S.data?.runs[runId(c)]
const now = () => performance.now() / 1000
/** training replay + prompt ingest are playing: dials and RUN are locked */
const locked = () => !!S.exp && !S.exp.stopped && ['train', 'ingest'].includes(phaseOf(S.exp, now()).phase)
/** the machine on the floor is the one the current experiment was recorded with */
const onMachine = () => !!S.exp && !S.exp.stopped && S.exp.run.id === runId(S.cfg)

// ---------------------------------------------------------------- actions

function setConfig(c: Config) {
  const prev = S.cfg
  // dropout only exists where variants were recorded
  if (!dropoutOptions(c, S.data?.meta).includes(c.dropout ?? 0)) c = { ...c, dropout: 0 }
  S.cfg = c
  // dropout changes the training, not the architecture: the machine stays put
  if (PARAMS.some(p => prev[p] !== c[p])) scene.setLayout(layoutFor(c, prev), now())
  // a different architecture is a different (untrained) machine; stop any generation in progress
  if (S.exp) S.exp.stopped = true
  if (S.selected && !scene.layout.parts.some(p => p.comp === S.selected) && S.selected !== 'terminal') S.selected = null
}

function step(p: Param, dir: number) {
  if (locked()) return
  const vals = grid()[p]
  const i = vals.indexOf(S.cfg[p]) + dir
  if (i < 0 || i >= vals.length) return
  setConfig({ ...S.cfg, [p]: vals[i] })
}

function stepDropout(dir: number) {
  if (locked()) return
  const vals = dropoutOptions(S.cfg, S.data?.meta)
  const i = vals.indexOf(S.cfg.dropout ?? 0) + dir
  if (i < 0 || i >= vals.length) return
  setConfig({ ...S.cfg, dropout: vals[i] })
}

async function runExperiment() {
  const run = runFor(S.cfg)
  if (!run || locked() || S.loading) return
  S.loading = true
  try {
    const detail = await loadDetail(run.id)
    if (run.id !== runId(S.cfg)) return // dial moved while loading
    const prev = S.exp && S.exp.run.id !== run.id ? S.exp.run : S.exp?.prev ?? null
    S.exp = { run, detail, prev, start: now(), genPos: 0, stopped: false }
    S.history = [...S.history.filter(h => h.id !== run.id), run]
    S.printoutOpen = S.notebookOpen = true
  } catch {
    S.loadError = `COULD NOT LOAD runs/${run.id}.json`
  } finally {
    S.loading = false
  }
}

function toggleSpeed() { S.realtime = !S.realtime }
function toggleScale() { S.scaleOpen = !S.scaleOpen; if (S.scaleOpen) S.selected = null }
function notebookShown() { return S.notebookOpen && S.history.length > 0 }
function printoutShown() { return !!S.exp && S.printoutOpen }
/** notebook (top) and printout (bottom) share a column on the right that the camera keeps clear */
function sidebarShown() { return notebookShown() || printoutShown() }

// ---------------------------------------------------------------- sizing + camera

function resize() {
  const dpr = window.devicePixelRatio || 1
  const devW = Math.round(window.innerWidth * dpr), devH = Math.round(window.innerHeight * dpr)
  S.pixel = Math.max(1, Math.round(devH / 500))
  const w = Math.ceil(devW / S.pixel), h = Math.ceil(devH / S.pixel)
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w
    canvas.height = h
    canvas.style.width = `${(w * S.pixel) / dpr}px`
    canvas.style.height = `${(h * S.pixel) / dpr}px`
    fb.resize(w, h)
  }
}

function screenExtent(boxes: Box[]) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
  for (const b of boxes) for (const dx of [0, b.w]) for (const dy of [0, b.d]) for (const dz of [0, b.h]) {
    const x = b.x + dx, y = b.y + dy, z = b.z + dz
    const sx = x - y, sy = (x + y) / 2 - z
    x0 = Math.min(x0, sx); x1 = Math.max(x1, sx); y0 = Math.min(y0, sy); y1 = Math.max(y1, sy)
  }
  return { x0, x1, y0, y1 }
}

function updateCamera(dt: number) {
  const e = screenExtent([scene.bounds(), FOREGROUND])
  const right = sidebarShown() ? NOTEBOOK_W + 12 : 8
  const left = 8 + Math.round(CONTROL.w * 0.45)
  const top = 14, bottom = 8
  const aw = fb.w - left - right, ah = fb.h - top - bottom
  const zoom = Math.min(aw / (e.x1 - e.x0), ah / (e.y1 - e.y0), 3)
  const tcx = (e.x0 + e.x1) / 2, tcy = (e.y0 + e.y1) / 2
  const c = S.cam
  if (!c.init) { c.zoom = zoom; c.sx = tcx; c.sy = tcy; c.init = true }
  const k = 1 - Math.exp(-dt * 3)
  c.zoom += (zoom - c.zoom) * k
  c.sx += (tcx - c.sx) * k
  c.sy += (tcy - c.sy) * k
  fb.zoom = c.zoom
  fb.cx = Math.round(left + aw / 2 - c.sx * c.zoom)
  fb.cy = Math.round(top + ah / 2 - c.sy * c.zoom)
}

/** screen anchor for a component's info leader line */
function anchorOf(comp: string): [number, number] | null {
  if (comp === 'terminal') return [fb.sx(CRT.x + CRT.w / 2, CRT.y + CRT.d), fb.sy(CRT.x + CRT.w / 2, CRT.y + CRT.d, CRT.z + CRT.h / 2)]
  let n = 0, ax = 0, ay = 0
  for (const l of scene.live.values()) {
    if (l.p.comp !== comp || l.dying !== null) continue
    const b = l.cur
    const x = b.x + b.w / 2, y = b.y + b.d, z = b.z + b.h / 2 + scene.bobNow
    ax += fb.sx(x, y); ay += fb.sy(x, y, z); n++
  }
  return n ? [Math.round(ax / n), Math.round(ay / n)] : null
}

function hoverLabel(): string {
  if (S.hoverUi) return S.hoverHint
  if (!S.hovered) return ''
  const shown = onMachine() ? machineState(S.exp, now(), S.data?.meta.vocab.length ?? 65).reveal : null
  if (S.hovered === 'lm' && S.hoverSub >= 0 && S.data) {
    const ch = S.data.meta.vocab[S.hoverSub]
    const label = ch === '\n' ? '↵' : ch === ' ' ? 'space' : ch
    const p = shown?.probs?.[S.hoverSub]
    return p !== undefined ? `'${label}'  p=${(p * 100).toFixed(1)}%` : `logit terminal '${label}'`
  }
  if (S.hovered === 'out' && S.hoverSub >= 0) {
    const out = shown?.output, ch = out?.[out.length - 1 - S.hoverSub]
    if (ch === undefined) return 'output belt'
    return `output '${ch === '\n' ? '↵' : ch === ' ' ? 'space' : ch}'  ${S.hoverSub === 0 ? 'newest' : `${S.hoverSub} back`}`
  }
  if (S.hovered === 'tok' && S.hoverSub >= 0) {
    const ch = shown?.context?.[S.hoverSub]
    return `position ${S.hoverSub}${ch !== undefined ? `  '${ch === '\n' ? '↵' : ch === ' ' ? 'space' : ch}'` : ''}`
  }
  return describe(S.hovered, S.cfg, null, [])?.title ?? ''
}

// ---------------------------------------------------------------- frame

let last = now()
function frame() {
  const t = now()
  const dt = t - last
  last = t
  resize()

  // resolve what is under the cursor using the previous frame's pick buffer
  S.hovered = null; S.hoverSub = -1; S.hoverUi = false; S.hoverHint = ''
  if (S.mx >= 0 && S.mx < fb.w && S.my >= 0 && S.my < fb.h) {
    const e = picker.get(fb.pick[S.my * fb.w + S.mx])
    if (e?.ui) { S.hoverUi = true; S.hoverHint = e.hint ?? '' }
    else if (e?.comp) { S.hovered = e.comp; S.hoverSub = e.sub }
  }
  canvas.style.cursor = S.hovered || S.hoverUi ? 'pointer' : 'default'

  if (S.exp) advance(S.exp, t, dt, S.realtime)
  const ms = onMachine() ? machineState(S.exp, t, S.data?.meta.vocab.length ?? 65) : machineState(null, t, 0)

  scene.update(dt, t)
  updateCamera(dt)

  picker.reset()
  resetGroups()
  const shellKey = `${fb.w}x${fb.h}@${fb.zoom.toFixed(5)},${fb.cx},${fb.cy}`
  if (!fb.restore(shellKey)) {
    fb.clear(P.ink)
    drawRoomShell(fb)
    fb.snapshot(shellKey)
  }
  drawRoom(fb, t, picker, ms.busy ? 1 : 0)
  const dctx = { t, picker, selected: S.selected, hovered: S.hovered, vocab: S.data?.meta.vocab ?? [], ...ms }
  scene.draw(fb, dctx)
  picker.mask(S.selected ?? S.hovered, hot)
  if (S.hovered && S.selected && S.hovered !== S.selected) {
    // keep the selection outlined while hovering something else
    picker.mask(S.selected, hot)
  }
  fb.outline(hot, P.lemon)
  scene.drawGlows(fb)
  scene.drawOverlays(fb, dctx)

  // ---- interfaces ----
  if (S.scaleOpen) {
    drawScale(fb, picker, S.cfg, S.data?.meta.recipe.max_iters ?? 3000, S.data?.meta.recipe.batch_size ?? 32, () => { S.scaleOpen = false })
  } else if (S.selected) {
    const rec = S.exp && S.exp.run.id === runId(S.cfg) && ms.reveal.probs ? { run: S.exp.run, detail: S.exp.detail } : null
    const info = describe(S.selected, S.cfg, rec, S.data?.meta.vocab ?? [])
    if (info) drawInfo(fb, picker, info, anchorOf(S.selected), () => { S.selected = null })
  } else {
    drawHint(fb, 'CLICK ANY PART  ·  ARROWS DIAL  ·  ENTER RUNS  ·  R SPEED  ·  G SCALE', 4, sidebarShown() ? NOTEBOOK_W + 12 : 0)
  }
  let avoid: { x: number; bottom: number } | null = null
  if (notebookShown()) {
    const busy = !!S.exp && !S.exp.stopped && phaseOf(S.exp, t).phase === 'train'
    avoid = drawNotebook(fb, picker, S.history, S.exp?.run.id ?? null, busy, t, r => setConfig({ ...r.config }), () => { S.notebookOpen = false })
  }
  const term = anchorOf('terminal')
  if (S.exp && printoutShown() && S.data && term) {
    const area = { x: fb.w - NOTEBOOK_W - 2, w: NOTEBOOK_W - 8, top: avoid ? avoid.bottom + 8 : 8 }
    drawPrintout(fb, picker, S.exp, t, S.data.meta, S.realtime, term, area, () => { S.printoutOpen = false })
  }
  drawControl(fb, picker, {
    cfg: S.cfg, row: S.row, grid: grid(), run: runFor(S.cfg), nRuns: S.data?.meta.n_runs ?? 0, totalRuns: total(),
    locked: locked() || S.loading, realtime: S.realtime,
    dropouts: dropoutOptions(S.cfg, S.data?.meta), stepDropout,
    step, setRow: i => { S.row = i }, runExperiment, toggleSpeed, toggleScale,
  }, t)
  if (S.loadError) drawHint(fb, S.loadError, fb.h - CONTROL.h - 20)
  const tip = hoverLabel()
  if (tip && !(S.hoverUi && !S.hoverHint)) drawTooltip(fb, S.mx, S.my, tip)

  ctx2d.putImageData(fb.img, 0, 0)
  requestAnimationFrame(frame)
}

// ---------------------------------------------------------------- input

function toPixel(ev: MouseEvent): [number, number] {
  const r = canvas.getBoundingClientRect()
  return [Math.floor(((ev.clientX - r.left) / r.width) * fb.w), Math.floor(((ev.clientY - r.top) / r.height) * fb.h)]
}

canvas.addEventListener('mousemove', ev => { [S.mx, S.my] = toPixel(ev) })
canvas.addEventListener('mouseleave', () => { S.mx = S.my = -1 })
canvas.addEventListener('mousedown', ev => {
  const [x, y] = toPixel(ev)
  const e = picker.get(fb.pick[y * fb.w + x])
  if (e?.ui) { e.ui(); return }
  if (e?.comp) { S.selected = e.comp; S.scaleOpen = false; return }
  S.selected = null
})

window.addEventListener('keydown', ev => {
  const k = ev.key
  const rows = PARAMS.length + 1 // the four architecture dials + dropout
  const dial = (dir: number) => (S.row < PARAMS.length ? step(PARAMS[S.row], dir) : stepDropout(dir))
  if (k === 'ArrowUp' || k === 'w') S.row = (S.row + rows - 1) % rows
  else if (k === 'ArrowDown' || k === 's') S.row = (S.row + 1) % rows
  else if (k === 'ArrowLeft' || k === 'a') dial(-1)
  else if (k === 'ArrowRight' || k === 'd') dial(1)
  else if (k === 'Enter' || k === ' ') runExperiment()
  else if (k === 'Escape') { S.selected = null; S.scaleOpen = false }
  else if (k === 'g' || k === 'G') toggleScale()
  else if (k === 'r' || k === 'R') toggleSpeed()
  else if (k >= '1' && k <= '5') S.row = Number(k) - 1
  else return
  ev.preventDefault()
})

// ---------------------------------------------------------------- boot

async function refresh() {
  try {
    S.data = await loadRuns()
    S.loadError = ''
  } catch {
    S.loadError = 'NO RECORDED RUNS FOUND - RUN lab/train/sweep.py'
  }
  if (!S.data || S.data.meta.n_runs < total()) setTimeout(refresh, 15000)
}

resize()
scene.setLayout(layoutFor(S.cfg), now())
refresh()
requestAnimationFrame(frame)
