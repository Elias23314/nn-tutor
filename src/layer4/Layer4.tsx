import { useMemo, useState } from 'react'
import { BrainTable, LayerHead, QuickCheck, RightIcon, SayItCard, WrongIcon } from '../components/LayerParts'
import { DEFAULT_ETA, DEFAULT_SEED, createNetwork, forward, neuronName, predict, type Network } from '../engine/network'
import { focusConnection, nudgeSlope } from '../layer3/slope'
import { connectionLabel, fmt, thousands, type Connection } from '../playground/connection'
import { useTrainer } from '../playground/useTrainer'
import { href } from '../routes'
import { deepConnection, gradients, outputChain, pathsFromPixel, type OutputChain } from './chain'
import '../playground/playground.css'
import '../components/layer.css'
import '../layer3/layer3.css'
import './layer4.css'

const LABEL = 7
type Factor = 1 | 2 | 3

export function Layer4() {
  const { status } = useTrainer()
  return (
    <main className="layer">
      <LayerHead
        n={4}
        name="The math"
        tag="Calc I"
        title="How does it get that number without nudging?"
        lede="The chain rule gives the steepness exactly. A connection into an output has one path to the mistake; deeper connections add up many."
        loopOn={['Find the slope']}
        loopCaption="This layer finds the slope exactly"
      />
      <QuickCheck
        before="It measures which direction makes the mistake "
        after="."
        options={['bigger', 'smaller', 'zero']}
        answer="smaller"
      />
      {status.kind === 'loading' && <p className="notice">Loading its training examples…</p>}
      {status.kind === 'error' && (
        <p className="notice" role="alert">
          Couldn&rsquo;t start the network: {status.message}
        </p>
      )}
      {status.kind === 'ready' && <Ready example={status.trainSamples[LABEL]} />}
    </main>
  )
}

function Ready({ example }: { example: Float32Array }) {
  const net = useMemo(() => createNetwork(undefined, DEFAULT_SEED), [])
  const c = useMemo(() => focusConnection(net, example, LABEL), [net, example])
  const ch = useMemo(() => outputChain(net, example, LABEL, c), [net, example, c])
  const name = connectionLabel(c)
  const source = neuronName(2, c.from)
  const [lit, setLit] = useState<Factor | null>(null)

  return (
    <>
      <Vocabulary />

      <div className="grid2 sec">
        <OnePath ch={ch} name={name} lit={lit} />
        <div className="stack">
          <ChainRule ch={ch} name={name} source={source} lit={lit} setLit={setLit} />
          <Update net={net} x={example} c={c} ch={ch} />
        </div>
      </div>

      <div className="sec">
        <CheckYourself ch={ch} />
      </div>

      <div className="sec">
        <DeepPaths net={net} x={example} />
      </div>

      <div className="sec">
        <Zeros net={net} x={example} />
      </div>

      <div className="sec">
        <YourTurn net={net} x={example} c={c} source={source} />
      </div>

      <div className="sec">
        <BrainTable
          topic="The learning rule"
          rows={[
            [
              'The error at each output is carried backward along every path',
              'Most known learning rules depend on activity near the synapse',
              'Nothing obviously carries an error backward like this. Feedback connections might approximate it; this is still an open question.',
            ],
          ]}
          so="So: the learning rule is where the brain comparison is least settled."
        />
      </div>

      <div className="sec">
        <SayItCard
          before="A connection’s gradient adds up, over every path to the mistake, the product of the local "
          answer="slopes"
          after=" along it."
          accepts={(t) => /slope|derivative|gradient|steep|rate/i.test(t)}
        />
      </div>

      <div className="exits">
        <a className="btn sec" href={href('layer-3')}>
          Back to layer 3
        </a>
        <a className="btn" href={href('playground')}>
          Try it in the playground
        </a>
      </div>
    </>
  )
}

const WORDS: [string, string, string][] = [
  ['strength', 'weight', 'w'],
  ['firing', 'activation', 'a'],
  ['mistake', 'loss', 'L'],
  ['steepness', 'gradient', '∂L/∂w'],
]

function Vocabulary() {
  return (
    <div className="panel">
      <div className="ptitle" style={{ flexWrap: 'wrap' }}>
        <span className="h3">Same ideas, formal names</span>
        <span className="small">Words from layers 1–3, then the term, then the symbol</span>
      </div>
      <div className="vocab">
        {WORDS.map(([word, term, sym]) => (
          <div className="vcard" key={word}>
            <span className="vw">{word}</span>
            <span className="va" aria-hidden="true">
              ↓
            </span>
            <span className="vt">
              {term} <span className="mono">{sym}</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

/** The single path from a hidden-layer-2 → output connection to the mistake. */
function OnePath({ ch, name, lit }: { ch: OutputChain; name: string; lit: Factor | null }) {
  const [dir, setDir] = useState<'forward' | 'backward'>('backward')
  const nodes = [
    { sym: 'w', value: fmt(ch.w, 4), cap: `strength of ${name}` },
    { sym: 'z₇', value: fmt(ch.z, 3), cap: 'weighted sum into 7' },
    { sym: 'ŷ₇', value: fmt(ch.yHat, 3), cap: '7 output' },
    { sym: 'L', value: fmt(ch.loss, 4), cap: 'mistake' },
  ]
  // Links between nodes i and i+1, numbered in the order the error signal reaches them.
  const links: { n: Factor; label: string }[] = [
    { n: 3, label: `∂z₇/∂w = a = ${fmt(ch.dzdw, 3)}` },
    { n: 2, label: `∂ŷ₇/∂z₇ = ŷ(1 − ŷ) = ${fmt(ch.dydz, 4)}` },
    { n: 1, label: `∂L/∂ŷ₇ = (2/10)(ŷ − 1) = ${fmt(ch.dLdy, 4)}` },
  ]
  const X = [70, 250, 430, 610]
  const Y = 150
  const back = dir === 'backward'
  return (
    <div className="panel">
      <div className="ptitle" style={{ flexWrap: 'wrap' }}>
        <span className="h3">One path to the mistake</span>
        <div className="chips" role="group" aria-label="Direction">
          {(['forward', 'backward'] as const).map((d) => (
            <button key={d} className="chip" aria-pressed={dir === d} onClick={() => setDir(d)}>
              {d === 'forward' ? 'Forward' : 'Backward'}
            </button>
          ))}
        </div>
      </div>
      <p className="body" style={{ marginBottom: 8 }}>
        From {name} there is only one route to the mistake: through the {LABEL} output.{' '}
        {back
          ? 'Going backward, each link contributes one local slope, numbered in the order the error signal reaches it.'
          : 'Going forward, each value is computed from the one before it.'}
      </p>
      <svg className="pathsvg" viewBox="0 0 680 240" role="img" aria-label={back ? `Backward: ${links.map((l) => l.label).join(', ')}` : `Forward: ${nodes.map((n) => `${n.sym} ${n.value}`).join(', ')}`}>
        <defs>
          {['ink', 'lit'].map((k) => (
            <marker key={k} id={`ah-${k}`} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0 0L8 4L0 8z" fill={k === 'lit' ? 'var(--excite)' : 'var(--ink)'} />
            </marker>
          ))}
        </defs>
        {links.map((l, i) => {
          const on = lit === l.n
          const x1 = X[i] + 34
          const x2 = X[i + 1] - 34
          return (
            <g key={l.n}>
              <line
                x1={back ? x2 : x1}
                y1={Y}
                x2={back ? x1 : x2}
                y2={Y}
                stroke={on ? 'var(--excite)' : 'var(--ink)'}
                strokeWidth={on ? 3.5 : 1.8}
                markerEnd={`url(#ah-${on ? 'lit' : 'ink'})`}
              />
              {back && (
                <>
                  <circle cx={(x1 + x2) / 2} cy={Y - 64 + (i % 2) * 32} r={11} fill={on ? 'var(--excite)' : 'var(--ink)'} />
                  <text x={(x1 + x2) / 2} y={Y - 60 + (i % 2) * 32} fontSize={12} fontWeight={600} fill="#fff" textAnchor="middle">
                    {l.n}
                  </text>
                  <text
                    x={(x1 + x2) / 2}
                    y={Y - 82 + (i % 2) * 32}
                    fontSize={12}
                    fill={on ? 'var(--excite)' : 'var(--ink-2)'}
                    textAnchor="middle"
                    fontFamily="var(--mono)"
                  >
                    {l.label}
                  </text>
                </>
              )}
            </g>
          )
        })}
        {nodes.map((n, i) => (
          <g key={n.sym}>
            <rect x={X[i] - 34} y={Y - 22} width={68} height={44} rx={8} fill="var(--card)" stroke="var(--ink)" strokeWidth={1.5} />
            <text x={X[i]} y={Y + 6} fontSize={17} fontWeight={600} textAnchor="middle" fill="var(--ink)">
              {n.sym}
            </text>
            <text x={X[i]} y={Y + 44} fontSize={13} textAnchor="middle" fill="var(--ink)" fontFamily="var(--mono)">
              {n.value}
            </text>
            <text x={X[i]} y={Y + 62} fontSize={12} textAnchor="middle" fill="var(--muted)">
              {n.cap}
            </text>
          </g>
        ))}
      </svg>
      <p className="small">
        {back
          ? 'Backward: the error signal starts at the mistake and is multiplied by each local slope on the way back.'
          : 'Forward: the strength sets the weighted sum, the sigmoid turns it into the output, and the output sets the mistake.'}
      </p>
    </div>
  )
}

function ChainRule({
  ch,
  name,
  source,
  lit,
  setLit,
}: {
  ch: OutputChain
  name: string
  source: string
  lit: Factor | null
  setLit: (f: Factor | null) => void
}) {
  const factors: { n: Factor; sym: string; work: string; value: number; digits: number; what: string }[] = [
    { n: 1, sym: '∂L/∂ŷ₇', work: `(2/10)(${fmt(ch.yHat, 3)} − 1)`, value: ch.dLdy, digits: 4, what: `How the mistake responds to the ${LABEL} output` },
    { n: 2, sym: '∂ŷ₇/∂z₇', work: `${fmt(ch.yHat, 3)} × ${fmt(1 - ch.yHat, 3)}`, value: ch.dydz, digits: 4, what: 'How the sigmoid output responds to its weighted sum' },
    { n: 3, sym: '∂z₇/∂w', work: 'a', value: ch.dzdw, digits: 3, what: `How the weighted sum responds to the strength: ${source}’s firing` },
  ]
  const hover = (n: Factor) => ({
    onMouseEnter: () => setLit(n),
    onMouseLeave: () => setLit(null),
    onFocus: () => setLit(n),
    onBlur: () => setLit(null),
    tabIndex: 0,
  })
  return (
    <div className="panel">
      <div className="ptitle" style={{ flexWrap: 'wrap' }}>
        <span className="h3">The chain rule for {name}</span>
        <span className="small">Hover a factor to light up its link</span>
      </div>
      <div className="formula mono">
        ∂L/∂w ={' '}
        {factors.map((f, i) => (
          <span key={f.n}>
            {i > 0 && ' · '}
            <span className={`fac${lit === f.n ? ' on' : ''}`} {...hover(f.n)}>
              {f.sym}
              <sup>{f.n}</sup>
            </span>
          </span>
        ))}
        <br />={' '}
        {factors.map((f) => (
          <span key={f.n} className={`fac${lit === f.n ? ' on' : ''}`} {...hover(f.n)}>
            ({fmt(f.value, f.digits)})
          </span>
        ))}{' '}
        = <b>{fmt(ch.gradient, 4)}</b>
      </div>
      <ol className="arith" style={{ marginTop: 10 }}>
        {factors.map((f) => (
          <li key={f.n} className={lit === f.n ? 'on' : undefined} {...hover(f.n)}>
            <span className="k">
              <span className="fnum">{f.n}</span> <span className="mono">{f.sym}</span>
            </span>
            <span className="v">
              <span className="mono">
                {f.work} = {fmt(f.value, f.digits)}
              </span>
              <br />
              <span className="small">{f.what}</span>
            </span>
          </li>
        ))}
      </ol>
    </div>
  )
}

function Update({ net, x, c, ch }: { net: Network; x: Float32Array; c: Connection; ch: OutputChain }) {
  const nudged = useMemo(() => nudgeSlope(net, x, LABEL, c, ch.w, 0.001), [net, x, c, ch.w])
  return (
    <div className="panel">
      <div className="h3" style={{ marginBottom: 10 }}>
        Update with learning rate {DEFAULT_ETA}
      </div>
      <p className="formula mono">
        w ← {fmt(ch.w, 4)} − {DEFAULT_ETA} × ({fmt(ch.gradient, 4)}) = {fmt(ch.w - DEFAULT_ETA * ch.gradient, 4)}
      </p>
      <p className="small" style={{ marginTop: 10 }}>
        Layer 3 measured <span className="mono">{fmt(nudged, 5)}</span> by nudging. The chain rule gives the exact value,{' '}
        <span className="mono">{fmt(ch.gradient, 5)}</span>.
      </p>
    </div>
  )
}

/** Put each factor on its link, with buttons. */
function CheckYourself({ ch }: { ch: OutputChain }) {
  const values = [
    { id: 3, text: fmt(ch.dzdw, 3) },
    { id: 1, text: fmt(ch.dLdy, 4) },
    { id: 2, text: fmt(ch.dydz, 4) },
  ]
  const links: { n: Factor; label: string }[] = [
    { n: 1, label: 'Link 1: mistake ← 7 output' },
    { n: 2, label: 'Link 2: 7 output ← weighted sum' },
    { n: 3, label: 'Link 3: weighted sum ← strength' },
  ]
  const [placed, setPlaced] = useState<Record<number, number>>({})
  const [checked, setChecked] = useState(false)
  const allPlaced = links.every((l) => placed[l.n] !== undefined)
  const score = links.filter((l) => placed[l.n] === l.n).length
  return (
    <div className="panel">
      <div className="ptitle" style={{ flexWrap: 'wrap' }}>
        <span className="h3">Check yourself</span>
        <span className="small">Put each factor on the link it belongs to</span>
      </div>
      <div className="place">
        {links.map((l) => (
          <div className="place-row" key={l.n} role="group" aria-label={l.label}>
            <span className="sub">{l.label}</span>
            <div className="chips">
              {values.map((v) => (
                <button
                  key={v.id}
                  className="chip mono"
                  aria-pressed={placed[l.n] === v.id}
                  disabled={checked}
                  onClick={() => setPlaced({ ...placed, [l.n]: v.id })}
                >
                  {v.text}
                  {checked && placed[l.n] === v.id && (v.id === l.n ? <RightIcon /> : <WrongIcon />)}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="btns" style={{ marginTop: 14 }}>
        {!checked ? (
          <button className="btn sm" disabled={!allPlaced} onClick={() => setChecked(true)}>
            Check
          </button>
        ) : (
          <button
            className="btn sec sm"
            onClick={() => {
              setChecked(false)
              setPlaced({})
            }}
          >
            Try again
          </button>
        )}
        {checked && (
          <span className="body" aria-live="polite">
            {score === 3
              ? 'All three right. The firing sits next to the strength, the sigmoid’s slope in the middle, the mistake’s slope at the end.'
              : `${score} of 3 right. The link nearest the mistake is how the mistake responds to the output: (2/10)(ŷ − 1).`}
          </span>
        )}
      </div>
    </div>
  )
}

function DeepPaths({ net, x }: { net: Network; x: Float32Array }) {
  const d = useMemo(() => deepConnection(net, x, LABEL), [net, x])
  const fwd = useMemo(() => forward(net, x), [net, x])
  const p = useMemo(() => pathsFromPixel(net, fwd, LABEL, d.from, d.to), [net, fwd, d])
  const live = p.paths.filter((q) => q.value !== 0)
  const top = [...live].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 5)
  const row = Math.floor(d.from / 28) + 1
  const col = (d.from % 28) + 1
  const j = neuronName(1, d.to)
  const sci = (v: number) => fmt(v * 1e6, 1)
  return (
    <div className="panel">
      <div className="ptitle" style={{ flexWrap: 'wrap' }}>
        <span className="h3">Deeper connections add up many paths</span>
        <span className="small">
          Pixel at row {row}, column {col} → neuron {j} in hidden layer 1
        </span>
      </div>
      <div className="deep">
        <div>
          <p className="body">
            From a pixel&rsquo;s connection, the error can travel back along 16 × 10 = 160 paths: through any neuron in
            hidden layer 2, then any of the ten outputs. The gradient is the sum, over every path, of the product of the
            local slopes along it.
          </p>
          <p className="formula mono" style={{ marginTop: 12 }}>
            ∂L/∂w = x · r₁ · Σₘ wₘ rₘ Σₖ wₘₖ ŷₖ(1 − ŷₖ) ∂L/∂ŷₖ
          </p>
          <p className="small" style={{ marginTop: 8 }}>
            r is ReLU&rsquo;s slope: 1 if the neuron fires, 0 if it is silent. Here x = {fmt(p.x, 2)} and r₁ = {p.r1}.
          </p>
          <div style={{ marginTop: 12 }}>
            <div className="kv">
              <span>Paths to the mistake</span>
              <span className="m">{p.paths.length}</span>
            </div>
            <div className="kv">
              <span>Through silent neurons, adding 0</span>
              <span className="m">{p.silentPaths}</span>
            </div>
            <div className="kv">
              <span>Sum of the other {p.paths.length - p.silentPaths}</span>
              <span className="m">{fmt(p.total, 6)}</span>
            </div>
            <p className="small" style={{ marginTop: 8 }}>
              That&rsquo;s {p.total > 0 ? '+' : ''}
              {sci(p.total)} millionths, exactly what backpropagation gives for this connection.
            </p>
          </div>
        </div>
        <div>
          <div className="sub" style={{ marginBottom: 6 }}>
            The five biggest paths, in millionths
          </div>
          <table className="ptable">
            <thead>
              <tr>
                <th>Through</th>
                <th>Then output</th>
                <th>Adds</th>
              </tr>
            </thead>
            <tbody>
              {top.map((q) => (
                <tr key={`${q.m}-${q.k}`}>
                  <td>{neuronName(2, q.m)}</td>
                  <td>{q.k}</td>
                  <td className="mono">{q.value > 0 ? `+${sci(q.value)}` : sci(q.value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small" style={{ marginTop: 10 }}>
            Paths pull both ways, and they partly cancel. Backpropagation never lists the paths. It works out each
            neuron&rsquo;s share of the error once, layer by layer, and reuses it, which is why it is fast.
          </p>
        </div>
      </div>
    </div>
  )
}

function Zeros({ net, x }: { net: Network; x: Float32Array }) {
  const { blankZero, silent, silentZero } = useMemo(() => {
    const g = gradients(net, x, LABEL).layers[0]
    const fwd = forward(net, x)
    let blankZero = 0
    for (let jj = 0; jj < g.outSize; jj++) if (g.weights[jj * g.inSize] === 0) blankZero++
    const silent = Array.from(fwd.layers[0].a).findIndex((v) => v === 0)
    let silentZero = 0
    if (silent >= 0) for (let i = 0; i < g.inSize; i++) if (g.weights[silent * g.inSize + i] === 0) silentZero++
    return { blankZero, silent, silentZero }
  }, [net, x])
  return (
    <div className="panel">
      <div className="h3" style={{ marginBottom: 12 }}>
        When a factor is zero
      </div>
      <div className="zeros">
        <div className="zcard">
          <div className="sub">Blank pixel</div>
          <p className="body">
            Row 1, column 1 has x = 0, so all {blankZero} of its connections get a gradient of 0.
          </p>
        </div>
        {silent >= 0 && (
          <div className="zcard">
            <div className="sub">Into neuron {neuronName(1, silent)}</div>
            <p className="body">
              Neuron {neuronName(1, silent)} in hidden layer 1 is silent for this {LABEL}, so ReLU&rsquo;s slope is 0 and
              all {thousands(silentZero)} connections into it get 0.
            </p>
          </div>
        )}
      </div>
      <p className="small" style={{ marginTop: 12 }}>
        These are layer 2&rsquo;s &ldquo;no change&rdquo; connections.
      </p>
    </div>
  )
}

const parse = (t: string) => Number(t.trim().replace('−', '-'))

function YourTurn({ net, x, c, source }: { net: Network; x: Float32Array; c: Connection; source: string }) {
  const guess = useMemo(() => predict(forward(net, x).yHat), [net, x])
  const wrong = guess === LABEL ? (LABEL + 1) % 10 : guess
  const ch = useMemo(() => outputChain(net, x, LABEL, { ...c, to: wrong }), [net, x, c, wrong])
  const [factor, setFactor] = useState('')
  const [result, setResult] = useState('')
  const [checked, setChecked] = useState<null | { f: boolean; r: boolean }>(null)
  return (
    <form
      className="panel turn"
      onSubmit={(e) => {
        e.preventDefault()
        setChecked({ f: Math.abs(parse(factor) - ch.dzdw) <= 0.01, r: Math.abs(parse(result) - ch.gradient) <= 0.0008 })
      }}
    >
      <div className="h3">Your turn</div>
      <p className="body">
        {source} → {wrong} leads to {guess === LABEL ? 'a wrong digit' : 'its wrong guess'}. What is its gradient?
      </p>
      <div className="turn-row mono" style={{ fontSize: 20 }}>
        ({fmt(ch.dLdy, 3)})({fmt(ch.dydz, 3)}) ×
        <input
          className="turn-box"
          inputMode="decimal"
          value={factor}
          placeholder="___"
          aria-label="Factor 3"
          onChange={(e) => {
            setFactor(e.target.value)
            setChecked(null)
          }}
        />
        =
        <input
          className="turn-box"
          inputMode="decimal"
          value={result}
          placeholder="___"
          aria-label="The gradient"
          onChange={(e) => {
            setResult(e.target.value)
            setChecked(null)
          }}
        />
        <button className="btn sec sm" disabled={!factor.trim() || !result.trim()} style={{ fontFamily: 'var(--sans)' }}>
          Check
        </button>
      </div>
      <p className="small">Hint: factor 3 is the same neuron&rsquo;s firing.</p>
      {checked && (
        <p className="feedback" aria-live="polite">
          {checked.f && checked.r
            ? `Right: ${fmt(ch.gradient, 4)}. It’s positive, so gradient descent weakens ${source} → ${wrong}, which lowers the wrong output.`
            : !checked.f
              ? `Not quite. Factor 3 is ${source}’s firing, ${fmt(ch.dzdw, 3)}, the same as for ${source} → ${LABEL}.`
              : `Factor 3 is right. Multiply all three: (${fmt(ch.dLdy, 3)})(${fmt(ch.dydz, 3)})(${fmt(ch.dzdw, 3)}) = ${fmt(ch.gradient, 4)}.`}
        </p>
      )}
    </form>
  )
}
