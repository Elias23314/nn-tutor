// Layer 1, step 2: the list of 784 numbers becomes the network's 784 inputs.
import { IMAGE_SIDE, PIXELS, neuronName, type ForwardResult, type Network } from '../engine/network'
import { fmt, thousands } from '../playground/connection'

const W = 720
const H = 420
const IMG = { x: 16, y: 140, size: 140 }
const IN_X = 360
const HID_X = 620
const hidY = (j: number) => 28 + 24 * j

const ink = (v: number) => {
  const s = 255 - Math.round(v * 255)
  return `rgb(${s},${s},${s})`
}
const shade = (t: number) => {
  const a = [0xd6, 0xd8, 0xdc]
  const b = [0x1f, 0x23, 0x28]
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * Math.min(1, t))).join(',')})`
}

type Row = { kind: 'input'; k: number } | { kind: 'gap'; skipped: number }

/** Which inputs to draw: the first two, the picked one and its neighbors, the last two, with gaps between. */
function rowsFor(pick: number): Row[] {
  const wanted = new Set<number>([0, 1, PIXELS - 2, PIXELS - 1])
  for (let d = -2; d <= 2; d++) if (pick + d >= 0 && pick + d < PIXELS) wanted.add(pick + d)
  const ks = [...wanted].sort((a, b) => a - b)
  const rows: Row[] = []
  ks.forEach((k, i) => {
    if (i > 0 && k - ks[i - 1] > 1) rows.push({ kind: 'gap', skipped: k - ks[i - 1] - 1 })
    rows.push({ kind: 'input', k })
  })
  return rows
}

export function InputLayer({
  image,
  network,
  fwd,
  pick,
}: {
  image: Float32Array
  network: Network
  fwd: ForwardResult
  pick: number
}) {
  const layer = network.layers[0]
  const a1 = fwd.layers[0].a
  const rows = rowsFor(pick)
  const step = Math.min(34, (H - 50) / rows.length)
  const rowY = (i: number) => 30 + step * i
  const inputY = new Map<number, number>()
  rows.forEach((r, i) => r.kind === 'input' && inputY.set(r.k, rowY(i)))

  const v = image[pick]
  const pushes = Array.from({ length: layer.outSize }, (_, j) => v * layer.weights[j * layer.inSize + pick])
  const maxPush = Math.max(...pushes.map(Math.abs), 1e-9)
  const maxA = Math.max(...a1, 1e-9)
  const row = Math.floor(pick / IMAGE_SIDE)
  const col = pick % IMAGE_SIDE
  const cell = IMG.size / IMAGE_SIDE

  return (
    <div className="inputs">
      <div className="inputs-scroll">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Input ${pick + 1} of 784, value ${fmt(v, 2)}, connected to all 16 neurons in the first layer.`}
        >
          <defs>
            <marker id="in-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0 0L8 4L0 8z" fill="var(--excite)" />
            </marker>
          </defs>

          {/* The picture, with the picked pixel's row marked: it is read row by row. */}
          {Array.from(image, (x, k) =>
            x > 0 ? (
              <rect key={k} x={IMG.x + (k % IMAGE_SIDE) * cell} y={IMG.y + Math.floor(k / IMAGE_SIDE) * cell} width={cell} height={cell} fill={ink(x)} />
            ) : null,
          )}
          <rect x={IMG.x} y={IMG.y} width={IMG.size} height={IMG.size} fill="none" stroke="#8E949C" />
          <rect x={IMG.x} y={IMG.y + row * cell} width={IMG.size} height={cell} fill="var(--excite)" fillOpacity={0.12} />
          <rect x={IMG.x + col * cell - 1} y={IMG.y + row * cell - 1} width={cell + 2} height={cell + 2} fill="none" stroke="var(--excite)" strokeWidth={2} />
          <text x={IMG.x + IMG.size / 2} y={IMG.y - 12} fontSize={12.5} fontWeight={500} fill="var(--muted)" textAnchor="middle">
            the picture
          </text>
          <text x={IMG.x + IMG.size / 2} y={IMG.y + IMG.size + 20} fontSize={12} fill="var(--muted)" textAnchor="middle">
            read row by row
          </text>
          <path
            d={`M ${IMG.x + (col + 1) * cell + 2} ${IMG.y + (row + 0.5) * cell} C ${IN_X - 170} ${IMG.y + (row + 0.5) * cell}, ${IN_X - 170} ${inputY.get(pick)}, ${IN_X - 124} ${inputY.get(pick)}`}
            fill="none"
            stroke="var(--excite)"
            strokeWidth={1.6}
            markerEnd="url(#in-head)"
          />

          {/* Every drawn input connects to every neuron; the picked input's 16 connections are colored. */}
          {rows.map((r) =>
            r.kind === 'input' && r.k !== pick
              ? Array.from({ length: layer.outSize }, (_, j) => (
                  <line key={`${r.k}-${j}`} x1={IN_X + 11} y1={inputY.get(r.k)} x2={HID_X - 10} y2={hidY(j)} stroke="#E4E3DE" strokeWidth={0.7} />
                ))
              : null,
          )}
          {pushes.map((p, j) => (
            <line
              key={`p${j}`}
              x1={IN_X + 11}
              y1={inputY.get(pick)}
              x2={HID_X - 10}
              y2={hidY(j)}
              stroke={v === 0 ? '#B9BCC1' : p > 0 ? 'var(--excite)' : 'var(--inhibit)'}
              strokeWidth={v === 0 ? 1 : 0.8 + 3 * Math.sqrt(Math.abs(p) / maxPush)}
              strokeDasharray={v === 0 || p < 0 ? '5 3' : undefined}
            />
          ))}

          {/* The input column. */}
          <text x={IN_X} y={14} fontSize={12.5} fontWeight={500} fill="var(--muted)" textAnchor="middle">
            784 inputs
          </text>
          {rows.map((r, i) =>
            r.kind === 'gap' ? (
              <text key={`g${i}`} x={IN_X - 18} y={rowY(i) + 4} fontSize={12} fill="var(--muted)" textAnchor="end">
                ⋮ {thousands(r.skipped)} more
              </text>
            ) : (
              <g key={r.k}>
                <circle
                  cx={IN_X}
                  cy={rowY(i)}
                  r={r.k === pick ? 11 : 9}
                  fill={ink(image[r.k])}
                  stroke={r.k === pick ? 'var(--excite)' : '#8E949C'}
                  strokeWidth={r.k === pick ? 2.5 : 1.2}
                />
                <text
                  x={IN_X - 18}
                  y={rowY(i) + 4}
                  fontSize={12}
                  textAnchor="end"
                  fontFamily="var(--mono)"
                  fontWeight={r.k === pick ? 600 : 400}
                  fill={r.k === pick ? 'var(--excite)' : 'var(--ink-3)'}
                >
                  #{thousands(r.k + 1)} = {fmt(image[r.k], 2)}
                </text>
              </g>
            ),
          )}

          {/* The first layer of neurons. What a neuron does with its inputs is step 3. */}
          <text x={HID_X} y={14} fontSize={12.5} fontWeight={500} fill="var(--muted)" textAnchor="middle">
            16 neurons
          </text>
          {Array.from(a1, (a, j) => (
            <g key={`n${j}`}>
              <circle cx={HID_X} cy={hidY(j)} r={9} fill={a > 0 ? shade(a / maxA) : '#fff'} stroke={a > 0 ? 'var(--ink)' : '#8E949C'} strokeWidth={1.4} />
              <text x={HID_X + 16} y={hidY(j) + 4} fontSize={12} fill="var(--muted)">
                {neuronName(1, j)}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div className="inputs-text">
        <p className="body">
          Each number in the list gets its own slot, called an <b>input</b>. The pixel you picked (row {row + 1}, column{' '}
          {col + 1}) is input <b>#{thousands(pick + 1)}</b>, and it holds <span className="mono">{fmt(v, 2)}</span>.
        </p>
        <p className="body">
          Every input is wired to all 16 neurons in the network&rsquo;s first layer. Each wire is a <b>connection</b>, and
          each connection has a <b>strength</b>: a number that decides how much of the input gets through, and whether it
          pushes that neuron up (blue) or down (orange, dashed).
        </p>
        <p className="body">
          {v === 0 ? (
            <>
              Your pixel is blank paper, so it sends <span className="mono">0</span> along its 16 connections. Whatever
              their strengths, it adds nothing.
            </>
          ) : (
            <>
              Your pixel sends its <span className="mono">{fmt(v, 2)}</span> along its 16 connections: thicker lines push
              harder.
            </>
          )}{' '}
          With 784 inputs and 16 neurons, that&rsquo;s 784 × 16 = {thousands(PIXELS * 16)} connections.
        </p>
      </div>
    </div>
  )
}
