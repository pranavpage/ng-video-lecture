// Per-frame registry mapping 16-bit pick-buffer ids to what was drawn there.

export interface PickEntry {
  comp: string // model component ('' for none)
  sub: number // e.g. which output terminal / cartridge
  ui?: () => void // screen-space button action
  hint?: string
}

export class Picker {
  private entries: PickEntry[] = [{ comp: '', sub: -1 }]
  private byKey = new Map<string, number>()

  reset() {
    this.entries.length = 1
    this.byKey.clear()
  }

  world(comp: string, sub = -1): number {
    if (!comp) return 0
    const k = comp + '|' + sub
    let id = this.byKey.get(k)
    if (id === undefined) {
      id = this.entries.length
      this.entries.push({ comp, sub })
      this.byKey.set(k, id)
    }
    return id
  }

  ui(action: () => void, hint?: string): number {
    this.entries.push({ comp: '', sub: -1, ui: action, hint })
    return this.entries.length - 1
  }

  get(id: number): PickEntry | undefined {
    return id > 0 ? this.entries[id] : undefined
  }

  /** flags every pick id that belongs to `comp` */
  mask(comp: string | null, out: Uint8Array) {
    out.fill(0)
    if (!comp) return
    for (let i = 1; i < this.entries.length; i++) if (this.entries[i].comp === comp) out[i] = 1
  }
}
