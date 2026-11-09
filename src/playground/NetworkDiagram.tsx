import { useMemo, useState, type ReactNode } from 'react'
import { hasInk } from '../engine/drawing'
import { IMAGE_SIDE, neuronName, predict, type ForwardResult, type Gradients, type Network } from '../engine/network'
import { connectionLabel, fmt, strength, type Connection } from './connection'

const TOP = 3
const IMG = { x: 20, y: 160, size: 120 }
const COLS = [300, 470, 640]
const hiddenY = (j: number) => 30 + 25 * j
const outputY = (k: number) => 72 + 33 * k

interface Line {
  c: Connection
  x1: number
  y1: number
  x2: number
  y2: number
  /** w · (what flowed through it): positive excites, negative inhibits. */
  push: number
}

function sourcePoint(c: Connection): [number, number] {
  if (c.layer === 0) {
    const cell = IMG.size / IMAGE_SIDE
    return [IMG.x + ((c.from % IMAGE_SIDE) + 0.5) * cell, IMG.y + (Math.floor(c.from / IMAGE_SIDE) + 0.5) * cell]
  }
  return [COLS[c.layer - 1] + 9, hiddenY(c.from)]
}

function targetPoint(c: Connection): [number, number] {
  return c.layer === 2 ? [COLS[2] - 13, outputY(c.to)] : [COLS[c.layer] - 9, hiddenY(c.to)]
}

function lineFor(net: Network, fwd: ForwardResult, c: Connection): Line {
  const l = net.layers[c.layer]
  const input = c.layer === 0 ? fwd.x : fwd.layers[c.layer - 1].a
  const [x1, y1] = sourcePoint(c)
  const [x2, y2] = targetPoint(c)
  return { c, x1, y1, x2, y2, push: l.weights[c.to * l.inSize + c.from] * input[c.from] }
}

/**
 * For every neuron, its `top` strongest incoming connections. By default "strongest" means
 * pushed hardest for this digit (w · a). With `byWeight`, hidden-to-hidden and hidden-to-output
 * connections are ranked by |w| instead, so connections from silent neurons show up too.
 * Pixel connections are always ranked by push, since blank pixels carry nothing.
 */
function strongestLines(net: Network, fwd: ForwardResult, top: number, byWeight: boolean): Line[] {
  const lines: Line[] = []
  net.layers.forEach((layer, l) => {
    const input = l === 0 ? fwd.x : fwd.layers[l - 1].a
    const rankByWeight = byWeight && l > 0
    for (let to = 0; to < layer.outSize; to++) {
      const ranked: [number, number][] = []
      for (let from = 0; from < layer.inSize; from++) {
        const w = layer.weights[to * layer.inSize + from]
        const push = w * input[from]
        if (rankByWeight) ranked.push([from, w])
        else if (push !== 0) ranked.push([from, push])
      }
      ranked.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      for (const [from] of ranked.slice(0, top)) lines.push(lineFor(net, fwd, { layer: l, from, to }))
    }
  })
  return lines
}

const sameConnection = (a: Connection, b: Connection) => a.layer === b.layer && a.from === b.from && a.to === b.to

const shade = (t: number) => {
  // From light gray (barely firing) to ink (firing hard).
  const a = [0xd6, 0xd8, 0xdc]
  const b = [0x1f, 0x23, 0x28]
  const mix = a.map((v, i) => Math.round(v + (b[i] - v) * Math.min(1, t)))
  return `rgb(${mix.join(',')})`
}

function imageUrl(x: ArrayLike<number>): string {
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = IMAGE_SIDE
  const ctx = canvas.getContext('2d')!
  const data = ctx.createImageData(IMAGE_SIDE, IMAGE_SIDE)
  for (let k = 0; k < IMAGE_SIDE * IMAGE_SIDE; k++) {
    const s = 255 - Math.round(x[k] * 255)
    data.data.set([s, s, s, 255], k * 4)
  }
  ctx.putImageData(data, 0, 0)
  return canvas.toDataURL()
}

interface Props {
  network: Network
  fwd: ForwardResult
  /** A connection to emphasize (and show even if it isn't among the strongest). */
  selected?: Connection | null
  /** Makes lines clickable. */
  onSelect?: (c: Connection) => void
  /** Connections shown into each neuron. */
  top?: number
  /**
   * The gradients of one correction. When given, connections the correction changes are
   * colored and the ones it leaves alone (gradient 0) are gray.
   */
  changed?: Gradients | null
  /** Panel heading; null for none. */
  title?: string | null
  caption: ReactNode
}

export function NetworkDiagram({ network, fwd, selected = null, onSelect, top = TOP, changed = null, title = 'The network', caption }: Props) {
  const [describe, setDescribe] = useState(false)
  const lines = useMemo(() => strongestLines(network, fwd, top, changed !== null), [network, fwd, top, changed])
  const selectedLine = selected ? lineFor(network, fwd, selected) : null
  const magnitude = (l: Line) => Math.abs(l.push) || Math.abs(strength(network, l.c)) * 0.5
  const maxMag = Math.max(...lines.map(magnitude), 1e-9)
  const url = useMemo(() => imageUrl(fwd.x), [fwd.x])
  const guess = predict(fwd.yHat)
  const isChanged = (c: Connection) => {
    if (!changed) return true
    const g = changed.layers[c.layer]
    return g.weights[c.to * g.inSize + c.from] !== 0
  }

  const width = (l: Line) => 0.8 + 2.8 * Math.sqrt(magnitude(l) / maxMag)
  const drawLine = (l: Line, key: string, emphasis = false) => {
    const gray = l.push === 0 || !isChanged(l.c)
    return (
      <g key={key}>
        {emphasis && <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} stroke="var(--ink)" strokeWidth={width(l) + 3} />}
        <line
          x1={l.x1}
          y1={l.y1}
          x2={l.x2}
          y2={l.y2}
          stroke={gray ? '#C9CBCF' : l.push > 0 ? 'var(--excite)' : 'var(--inhibit)'}
          strokeWidth={width(l)}
          strokeOpacity={emphasis ? 1 : 0.85}
          strokeDasharray={!gray && l.push < 0 ? '5 3' : undefined}
        />
        <line
          className={onSelect ? 'hit' : undefined}
          x1={l.x1}
          y1={l.y1}
          x2={l.x2}
          y2={l.y2}
          stroke="transparent"
          strokeWidth={10}
          onClick={onSelect && (() => onSelect(l.c))}
        >
          <title>{`${connectionLabel(l.c)}: ${changed ? (isChanged(l.c) ? 'changes in this correction' : 'unchanged') : `pushes ${fmt(l.push)}`}`}</title>
        </line>
      </g>
    )
  }

  const hidden = [fwd.layers[0].a, fwd.layers[1].a]
  const firingCounts = hidden.map((a) => Array.from(a).filter((v) => v > 0).length)
  const ranked = Array.from(fwd.yHat)
    .map((v, d) => [d, v] as const)
    .sort((a, b) => b[1] - a[1])

  return (
    <div className="panel net">
      <div className="ptitle">
        <span className="h3">{title ?? ''}</span>
        <button className="tog" role="switch" aria-checked={describe} onClick={() => setDescribe(!describe)}>
          <span className="sw">
            <i />
          </span>
          Describe this
        </button>
      </div>
      <svg viewBox="0 0 760 450" role="img" aria-label="Diagram of the network reading your digit">
        <image href={url} x={IMG.x} y={IMG.y} width={IMG.size} height={IMG.size} style={{ imageRendering: 'pixelated' }} />
        <rect x={IMG.x} y={IMG.y} width={IMG.size} height={IMG.size} fill="none" stroke="#8E949C" />
        {/* Gray (unchanged) lines go underneath the colored ones. */}
        {lines
          .filter((l) => !selected || !sameConnection(l.c, selected))
          .sort((a, b) => Number(isChanged(a.c) && a.push !== 0) - Number(isChanged(b.c) && b.push !== 0))
          .map((l, i) => drawLine(l, `l${i}`))}
        {selectedLine && drawLine(selectedLine, 'selected', true)}

        {hidden.map((a, h) => {
          const max = Math.max(...a, 1e-9)
          return Array.from(a, (v, j) => (
            <circle
              key={`${h}-${j}`}
              cx={COLS[h]}
              cy={hiddenY(j)}
              r={9}
              fill={v > 0 ? shade(v / max) : '#fff'}
              stroke={v > 0 ? 'var(--ink)' : '#8E949C'}
              strokeWidth={1.4}
            >
              <title>{`Neuron ${neuronName(h === 0 ? 1 : 2, j)} in hidden layer ${h + 1}: ${v > 0 ? `firing ${fmt(v, 2)}` : 'silent'}`}</title>
            </circle>
          ))
        })}
        {Array.from(fwd.yHat, (v, k) => (
          <g key={k}>
            <circle cx={COLS[2]} cy={outputY(k)} r={13} fill={shade(v)} stroke="var(--ink)" strokeWidth={1.4} />
            <text
              x={COLS[2]}
              y={outputY(k) + 4.5}
              fontSize={13}
              fontWeight={600}
              textAnchor="middle"
              fill={v > 0.55 ? '#fff' : 'var(--ink)'}
              pointerEvents="none"
            >
              {k}
            </text>
            {k === guess && (
              <text x={COLS[2] + 22} y={outputY(k) + 4.5} fontSize={12} fontWeight={500} fill="var(--muted)">
                its guess
              </text>
            )}
          </g>
        ))}
        {[
          [IMG.x + IMG.size / 2, IMG.y + IMG.size + 22, '784 pixels'],
          [COLS[0], 438, 'hidden layer 1'],
          [COLS[1], 438, 'hidden layer 2'],
          [COLS[2], 438, '10 outputs'],
        ].map(([x, y, label]) => (
          <text key={label} x={x} y={y} fontSize={12.5} fontWeight={500} fill="var(--muted)" textAnchor="middle">
            {label}
          </text>
        ))}
      </svg>
      <div className="legend">
        <span className="i">
          <svg width="34" height="12" aria-hidden="true">
            <line x1="1" y1="6" x2="33" y2="6" stroke="var(--excite)" strokeWidth="3" />
          </svg>
          excites
        </span>
        <span className="i">
          <svg width="34" height="12" aria-hidden="true">
            <line x1="1" y1="6" x2="33" y2="6" stroke="var(--inhibit)" strokeWidth="3" strokeDasharray="5 3" />
          </svg>
          inhibits (dashed)
        </span>
        <span className="i">
          <svg width="16" height="16" aria-hidden="true">
            <circle cx="8" cy="8" r="6.5" fill="#2B2F35" stroke="var(--ink)" strokeWidth="1.5" />
          </svg>
          firing, darker is more
        </span>
        <span className="i">
          <svg width="16" height="16" aria-hidden="true">
            <circle cx="8" cy="8" r="6.5" fill="#fff" stroke="#8E949C" strokeWidth="1.5" />
          </svg>
          silent
        </span>
      </div>
      {changed && (
        <div className="legend" style={{ marginTop: 6 }}>
          <span className="i">
            <svg width="34" height="12" aria-hidden="true">
              <line x1="1" y1="6" x2="33" y2="6" stroke="#C9CBCF" strokeWidth="3" />
            </svg>
            doesn&rsquo;t change
          </span>
        </div>
      )}
      <p className="small" style={{ marginTop: 10 }}>
        {caption}
      </p>
      {describe && !hasInk(fwd.x) && (
        <p className="describe">
          The pad is blank, so no pixel sends anything in. What you see is the network running on biases alone: its
          strongest output is {ranked[0][0]} ({fmt(ranked[0][1], 2)}).
        </p>
      )}
      {describe && hasInk(fwd.x) && (
        <p className="describe">
          This digit makes {firingCounts[0]} of the 16 neurons in hidden layer 1 fire, and {firingCounts[1]} of the 16 in
          hidden layer 2. Its strongest output is {ranked[0][0]} ({fmt(ranked[0][1], 2)}), then {ranked[1][0]} (
          {fmt(ranked[1][1], 2)}).{' '}
          {changed &&
            `This correction changes the connections into the ${firingCounts[0]} firing neurons of hidden layer 1 from inked pixels, and leaves every connection that starts or ends at a silent neuron alone.`}
          {selected && selectedLine && (
            <>
              {connectionLabel(selected)} has strength {fmt(strength(network, selected))} and{' '}
              {selectedLine.push === 0
                ? 'carries nothing for this digit, because its source is silent.'
                : `${selectedLine.push > 0 ? 'excites' : 'inhibits'} its target by ${fmt(Math.abs(selectedLine.push))}.`}
            </>
          )}
        </p>
      )}
    </div>
  )
}
