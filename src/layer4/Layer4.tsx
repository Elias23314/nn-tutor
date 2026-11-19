import { useMemo, useState, type ReactNode } from 'react'
import { BrainTable, LayerHead, QuickCheck, RightIcon, SayItCard, WrongIcon } from '../components/LayerParts'
import { DEFAULT_ETA, DEFAULT_SEED, createNetwork, forward, neuronName, predict, type Network } from '../engine/network'
import { focusConnection, nudgeSlope } from '../layer3/slope'
import { connectionLabel, fmt, thousands, type Connection } from '../playground/connection'
import { NetworkDiagram, type Mark } from '../playground/NetworkDiagram'
import { useTrainer } from '../playground/useTrainer'
import { href } from '../routes'
import { deepConnection, gradients, outputChain, pathsFromPixel, type OutputChain } from './chain'
import '../playground/playground.css'
import '../components/layer.css'
import '../layer3/layer3.css'
import './layer4.css'

const LABEL = 7
type Factor = 1 | 2 | 3
/** What the page is pointing at: a walkthrough step, or a factor you're hovering. */
type Focus = 'want' | 'forward' | Factor | 'all' | 'step'

export function Layer4() {
  const { status } = useTrainer()
  return (
    <main className="layer">
      <LayerHead
        n={4}
        name="The math"
        tag="Calculus"
        title="How does it get that number without nudging?"
        soFar="For one connection, the slope says which way to move and the learning rate says how far. In part 3 you found the slope by nudging the connection and watching the mistake."
        lede="Nudging 13,002 connections one at a time would be far too slow. The chain rule gives the slope exactly. A connection into an output has one path to the mistake; deeper connections add up many."
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
  const [step, setStep] = useState(0)
  const [hover, setHover] = useState<Factor | null>(null)
  const fwd = useMemo(() => forward(net, example), [net, example])
  const steps = walkSteps(ch, name, source)
  const focus: Focus = hover ?? steps[step].focus
  const after = ch.w - DEFAULT_ETA * ch.gradient
  // Where each part of the formula lives in the network.
  const marks: Mark[] =
    focus === 'forward'
      ? [
          { layer: 2, index: c.from, label: `a = ${fmt(ch.a, 2)}` },
          { layer: 'out', index: LABEL, label: `ŷ₇ = ${fmt(ch.yHat, 3)}` },
        ]
      : focus === 1
        ? [{ layer: 'out', index: LABEL, label: `ŷ₇ = ${fmt(ch.yHat, 3)}\ngoal 1` }]
        : focus === 2
          ? [{ layer: 'out', index: LABEL, label: `sigmoid\nslope ${fmt(ch.dydz, 3)}` }]
          : focus === 3
            ? [{ layer: 2, index: c.from, label: `${source} fires ${fmt(ch.a, 2)}` }]
            : focus === 'all'
              ? [
                  { layer: 2, index: c.from, label: `× ${fmt(ch.dzdw, 3)}` },
                  { layer: 'out', index: LABEL, label: `× ${fmt(ch.dydz, 3)}\n× (${fmt(ch.dLdy, 4)})` },
                ]
              : focus === 'step'
                ? [
                    { layer: 2, index: c.from },
                    { layer: 'out', index: LABEL, label: `w: ${fmt(ch.w, 3)}\n→ ${fmt(after, 3)}` },
                  ]
                : [
                    { layer: 2, index: c.from },
                    { layer: 'out', index: LABEL },
                  ]
  const caption =
    focus === 'want'
      ? `The outlined line is ${name}: its weight is w. ∂L/∂w asks how much the mistake changes when w changes a little.`
      : focus === 'forward'
        ? `Forward: ${source}’s firing flows along ${name} into the ${LABEL} output, and the outputs set the mistake.`
        : focus === 1
          ? `Factor 1 lives at the ${LABEL} output: how far it is from its goal of 1.`
          : focus === 2
            ? `Factor 2 also lives in the ${LABEL} output: how steeply its sigmoid responds to the weighted sum coming in.`
            : focus === 3
              ? `Factor 3 lives at neuron ${source}: how hard it fires, which is what flows along ${name}.`
              : focus === 'all'
                ? `Backward along the path: from the mistake, through the ${LABEL} output, along ${name} to ${source}, multiplying as it goes.`
                : `The step: the weight of ${name} goes up, from ${fmt(ch.w, 3)} to ${fmt(after, 3)}.`

  return (
    <>
      <Vocabulary />

      <div className="grid2 sec">
        <div className="sticky-col">
          <NetworkDiagram
            title={`${name} in the network`}
            network={net}
            fwd={fwd}
            selected={c}
            marks={marks}
            dimOthers
            caption={<span aria-live="polite">{caption}</span>}
          />
        </div>
        <div className="stack">
          <Walkthrough steps={steps} step={step} setStep={setStep} />
          <ChainRule ch={ch} name={name} source={source} focus={focus} setHover={setHover} />
          <OnePath ch={ch} name={name} focus={focus} setHover={setHover} />
          <Update net={net} x={example} c={c} ch={ch} on={focus === 'step'} />
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
          before="Backpropagation sends the mistake backward through the network: each connection’s gradient adds up, over every path to the mistake, the product of the local "
          answer="slopes"
          after=" along it."
          accepts={(t) => /slope|derivative|gradient|steep|rate/i.test(t)}
          next={{
            label: 'The playground',
            text: 'That’s the whole loop: predict, measure the mistake, find every slope with backpropagation, step, and repeat thousands of times. In the playground you can run that loop yourself: train the network, draw your own digits, change its examples, and try to make it fail.',
          }}
        />
      </div>

      <div className="exits">
        <a className="btn sec" href={href('layer-3')}>
          Back to part 3
        </a>
        <a className="btn" href={href('playground')}>
          Try it in the playground
        </a>
      </div>
    </>
  )
}

interface WalkStep {
  title: string
  /** Which stages of the learning loop this step belongs to. */
  stages: string[]
  focus: Focus
  formula: ReactNode
  text: ReactNode
}

function walkSteps(ch: OutputChain, name: string, source: string): WalkStep[] {
  const after = ch.w - DEFAULT_ETA * ch.gradient
  const n = (v: number, d = 3) => <span className="mono">{fmt(v, d)}</span>
  return [
    {
      title: 'The question',
      stages: ['Find the slope'],
      focus: 'want',
      formula: '∂L/∂w = ?',
      text: (
        <>
          Read <span className="mono">∂L/∂w</span> as &ldquo;how much does the mistake L change when the weight w of{' '}
          {name} changes a little?&rdquo; It&rsquo;s the slope you measured in part 3 by nudging. The chain rule gets
          it exactly, by following the route from w to L through the network.
        </>
      ),
    },
    {
      title: 'First, the forward pass',
      stages: ['Guess', 'Measure the mistake'],
      focus: 'forward',
      formula: 'w → z₇ = … + w·a → ŷ₇ = σ(z₇) → L',
      text: (
        <>
          {source} fires a = {n(ch.a)}. That flows along {name}, multiplied by its weight w, into the {LABEL}{' '}
          output&rsquo;s weighted sum z₇ = {n(ch.z)}. The sigmoid σ squeezes it into the output ŷ₇ = {n(ch.yHat)}. The
          mistake compares all ten outputs with their goals: L = {n(ch.loss, 4)}. So w reaches the mistake in three links:
          w → z₇ → ŷ₇ → L.
        </>
      ),
    },
    {
      title: 'Factor 1: the mistake’s end',
      stages: ['Find the slope'],
      focus: 1,
      formula: `∂L/∂ŷ₇ = (2/10)(ŷ₇ − 1) = ${fmt(ch.dLdy, 4)}`,
      text: (
        <>
          Start at the mistake and work backward. If the {LABEL} output rose a little, how would the mistake change? The
          answer is negative, so raising ŷ₇ shrinks the mistake: it&rsquo;s at {n(ch.yHat)}, below its goal of 1. The 2/10
          comes from averaging ten squared gaps.
        </>
      ),
    },
    {
      title: 'Factor 2: through the output',
      stages: ['Find the slope'],
      focus: 2,
      formula: `∂ŷ₇/∂z₇ = ŷ₇(1 − ŷ₇) = ${fmt(ch.dydz, 4)}`,
      text: (
        <>
          Step back into the {LABEL} output. Its sigmoid turns the weighted sum z₇ into a number between 0 and 1, and its
          slope is ŷ(1 − ŷ). At ŷ₇ = {n(ch.yHat)} that&rsquo;s {n(ch.dydz)}, close to as steep as a sigmoid gets: a change in
          z₇ comes out about a quarter as big in ŷ₇.
        </>
      ),
    },
    {
      title: `Factor 3: along ${name}`,
      stages: ['Find the slope'],
      focus: 3,
      formula: `∂z₇/∂w = a = ${fmt(ch.dzdw, 3)}`,
      text: (
        <>
          Step back along {name} to neuron {source}. The weighted sum includes w × a, so nudging w changes z₇ by a times as
          much: {source}&rsquo;s firing, {n(ch.a)}. If {source} were silent (a = 0), this factor, and the whole gradient,
          would be 0. Those are part 2&rsquo;s &ldquo;no change&rdquo; connections.
        </>
      ),
    },
    {
      title: 'Multiply: the chain rule',
      stages: ['Find the slope'],
      focus: 'all',
      formula: `∂L/∂w = (${fmt(ch.dLdy, 4)})(${fmt(ch.dydz, 4)})(${fmt(ch.dzdw, 3)}) = ${fmt(ch.gradient, 4)}`,
      text: (
        <>
          Each factor turns one small change into the next: a change in w becomes {n(ch.dzdw)}× as big in z₇, then{' '}
          {n(ch.dydz)}× that in ŷ₇, then {n(ch.dLdy, 4)}× that in L. Chaining them gives the gradient, {n(ch.gradient, 4)}:
          the same slope part 3 found by nudging, but exact.
        </>
      ),
    },
    {
      title: 'Use it: take the step',
      stages: ['Take a step', 'Repeat'],
      focus: 'step',
      formula: `w ← w − η · ∂L/∂w = ${fmt(ch.w, 4)} − ${DEFAULT_ETA} × (${fmt(ch.gradient, 4)}) = ${fmt(after, 4)}`,
      text: (
        <>
          The gradient is negative, so subtracting it raises the weight of {name}, which lifts the {LABEL} output toward its goal.
          η is the learning rate from part 3. Backpropagation does this for all 13,002 connections at once, then the loop
          repeats with the next examples.
        </>
      ),
    },
  ]
}

/** Reads the formula one piece at a time, pointing at the network for each. */
function Walkthrough({ steps, step, setStep }: { steps: WalkStep[]; step: number; setStep: (s: number) => void }) {
  const s = steps[step]
  return (
    <div className="panel walk">
      <div className="ptitle" style={{ flexWrap: 'wrap' }}>
        <span className="h3">Read the formula, one piece at a time</span>
        <span className="small">
          Step {step + 1} of {steps.length}
        </span>
      </div>
      <ol className="walk-dots" aria-label="Steps">
        {steps.map((t, i) => (
          <li key={t.title}>
            <button
              className={i === step ? 'on' : i < step ? 'done' : undefined}
              aria-current={i === step ? 'step' : undefined}
              aria-label={`Step ${i + 1}: ${t.title}`}
              onClick={() => setStep(i)}
            >
              {i + 1}
            </button>
          </li>
        ))}
      </ol>
      <div className="walk-body" aria-live="polite">
        <div className="walk-stage">
          <span className="small">In the learning loop:</span>
          {s.stages.map((st) => (
            <span key={st} className="stage-pill">
              {st}
            </span>
          ))}
        </div>
        <div className="h3 walk-title">{s.title}</div>
        <div className="walk-formula mono">{s.formula}</div>
        <p className="body">{s.text}</p>
      </div>
      <div className="btns" style={{ marginTop: 14 }}>
        <button className="btn sec sm" disabled={step === 0} onClick={() => setStep(step - 1)}>
          Back
        </button>
        {step < steps.length - 1 ? (
          <button className="btn sm" onClick={() => setStep(step + 1)}>
            Next: {steps[step + 1].title}
          </button>
        ) : (
          <button className="btn sm" onClick={() => setStep(0)}>
            Start again
          </button>
        )}
      </div>
    </div>
  )
}

const WORDS: [string, string, string][] = [
  ['weight', 'weight (already the formal word)', 'w'],
  ['firing', 'activation', 'a'],
  ['mistake', 'loss', 'L'],
  ['slope', 'gradient', '∂L/∂w'],
]

function Vocabulary() {
  return (
    <div className="panel">
      <div className="ptitle" style={{ flexWrap: 'wrap' }}>
        <span className="h3">Same ideas, formal names</span>
        <span className="small">Words from parts 1–3, then the term, then the symbol</span>
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
function OnePath({
  ch,
  name,
  focus,
  setHover,
}: {
  ch: OutputChain
  name: string
  focus: Focus
  setHover: (f: Factor | null) => void
}) {
  const [chosen, setDir] = useState<'forward' | 'backward'>('backward')
  // The walkthrough's forward step shows the forward view; its factor steps show the backward one.
  const dir = focus === 'forward' ? 'forward' : typeof focus === 'number' || focus === 'all' ? 'backward' : chosen
  const nodes = [
    { sym: 'w', value: fmt(ch.w, 4), cap: `weight of ${name}` },
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
          const on = focus === l.n || focus === 'all'
          const x1 = X[i] + 34
          const x2 = X[i + 1] - 34
          return (
            <g key={l.n} onMouseEnter={() => setHover(l.n)} onMouseLeave={() => setHover(null)} style={{ cursor: 'default' }}>
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
          : 'Forward: the weight sets the weighted sum, the sigmoid turns it into the output, and the output sets the mistake.'}
      </p>
    </div>
  )
}

function ChainRule({
  ch,
  name,
  source,
  focus,
  setHover,
}: {
  ch: OutputChain
  name: string
  source: string
  focus: Focus
  setHover: (f: Factor | null) => void
}) {
  const isOn = (n: Factor) => focus === n || focus === 'all'
  const factors: { n: Factor; sym: string; work: string; value: number; digits: number; what: string }[] = [
    { n: 1, sym: '∂L/∂ŷ₇', work: `(2/10)(${fmt(ch.yHat, 3)} − 1)`, value: ch.dLdy, digits: 4, what: `How the mistake responds to the ${LABEL} output` },
    { n: 2, sym: '∂ŷ₇/∂z₇', work: `${fmt(ch.yHat, 3)} × ${fmt(1 - ch.yHat, 3)}`, value: ch.dydz, digits: 4, what: 'How the sigmoid output responds to its weighted sum' },
    { n: 3, sym: '∂z₇/∂w', work: 'a', value: ch.dzdw, digits: 3, what: `How the weighted sum responds to the weight: ${source}’s firing` },
  ]
  const hover = (n: Factor) => ({
    onMouseEnter: () => setHover(n),
    onMouseLeave: () => setHover(null),
    onFocus: () => setHover(n),
    onBlur: () => setHover(null),
    tabIndex: 0,
  })
  return (
    <div className="panel">
      <div className="ptitle" style={{ flexWrap: 'wrap' }}>
        <span className="h3">The chain rule for {name}</span>
        <span className="small">Hover a factor to light up its link</span>
      </div>
      <div className="formula mono">
        <span className={`fac${focus === 'want' ? ' on' : ''}`}>∂L/∂w</span> ={' '}
        {factors.map((f, i) => (
          <span key={f.n}>
            {i > 0 && ' · '}
            <span className={`fac${isOn(f.n) ? ' on' : ''}`} {...hover(f.n)}>
              {f.sym}
              <sup>{f.n}</sup>
            </span>
          </span>
        ))}
        <br />={' '}
        {factors.map((f) => (
          <span key={f.n} className={`fac${isOn(f.n) ? ' on' : ''}`} {...hover(f.n)}>
            ({fmt(f.value, f.digits)})
          </span>
        ))}{' '}
        = <b className={`fac${focus === 'all' ? ' on' : ''}`}>{fmt(ch.gradient, 4)}</b>
      </div>
      <ol className="arith" style={{ marginTop: 10 }}>
        {factors.map((f) => (
          <li key={f.n} className={isOn(f.n) ? 'on' : undefined} {...hover(f.n)}>
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

function Update({ net, x, c, ch, on }: { net: Network; x: Float32Array; c: Connection; ch: OutputChain; on: boolean }) {
  const nudged = useMemo(() => nudgeSlope(net, x, LABEL, c, ch.w, 0.001), [net, x, c, ch.w])
  return (
    <div className={`panel${on ? ' panel-on' : ''}`}>
      <div className="h3" style={{ marginBottom: 10 }}>
        Update with learning rate {DEFAULT_ETA}
      </div>
      <p className="formula mono">
        w ← {fmt(ch.w, 4)} − {DEFAULT_ETA} × ({fmt(ch.gradient, 4)}) = {fmt(ch.w - DEFAULT_ETA * ch.gradient, 4)}
      </p>
      <p className="small" style={{ marginTop: 10 }}>
        Part 3 measured <span className="mono">{fmt(nudged, 5)}</span> by nudging. The chain rule gives the exact value,{' '}
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
    { n: 3, label: 'Link 3: weighted sum ← weight' },
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
              ? 'All three right. The firing sits next to the weight, the sigmoid’s slope in the middle, the mistake’s slope at the end.'
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
  const top = useMemo(
    () =>
      p.paths
        .filter((q) => q.value !== 0)
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 5),
    [p],
  )
  const [picked, setPicked] = useState(0)
  const path = top[picked]
  const row = Math.floor(d.from / 28) + 1
  const col = (d.from % 28) + 1
  const j = neuronName(1, d.to)
  const m = neuronName(2, path.m)
  const sci = (v: number) => fmt(v * 1e6, 1)
  const signedSci = (v: number) => (v > 0 ? `+${sci(v)}` : sci(v))

  // The local slopes along the picked path, left to right, as they appear in the formula.
  const w1 = net.layers[1].weights[path.m * net.layers[1].inSize + d.to]
  const w2 = net.layers[2].weights[path.k * net.layers[2].inSize + path.m]
  const y = fwd.yHat[path.k]
  const dLdy = (2 / 10) * (y - (path.k === LABEL ? 1 : 0))
  const factors: [string, number, number][] = [
    ['x', p.x, 2],
    ['r₁', p.r1, 0],
    [`w ${j}→${m}`, w1, 3],
    ['r₂', 1, 0],
    [`w ${m}→${path.k}`, w2, 3],
    [`ŷ${path.k}(1 − ŷ${path.k})`, y * (1 - y), 3],
    [`∂L/∂ŷ${path.k}`, dLdy, 3],
  ]

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
              That&rsquo;s {signedSci(p.total)} millionths, exactly what backpropagation gives for this connection.
            </p>
          </div>

          <div className="sub" style={{ margin: '20px 0 6px' }}>
            The five biggest paths, in millionths: click one to draw it
          </div>
          <table className="ptable paths">
            <thead>
              <tr>
                <th>Through</th>
                <th>Then output</th>
                <th>Adds</th>
              </tr>
            </thead>
            <tbody>
              {top.map((q, i) => (
                <tr
                  key={`${q.m}-${q.k}`}
                  className={i === picked ? 'on' : undefined}
                  tabIndex={0}
                  aria-selected={i === picked}
                  onClick={() => setPicked(i)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      setPicked(i)
                    }
                  }}
                >
                  <td>{neuronName(2, q.m)}</td>
                  <td>{q.k}</td>
                  <td className="mono">{signedSci(q.value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="small" style={{ marginTop: 10 }}>
            Paths pull both ways, and they partly cancel. Backpropagation never lists the paths. It works out each
            neuron&rsquo;s share of the error once, layer by layer, and reuses it, which is why it is fast.
          </p>
        </div>

        <div className="stack">
          <NetworkDiagram
            title={null}
            network={net}
            fwd={fwd}
            selected={d}
            extra={[
              { layer: 1, from: d.to, to: path.m },
              { layer: 2, from: path.m, to: path.k },
            ]}
            marks={[
              { layer: 1, index: d.to },
              { layer: 2, index: path.m },
              { layer: 'out', index: path.k, label: path.k === LABEL ? 'goal 1' : 'goal 0' },
            ]}
            dimOthers
            caption={`One of the 160 paths: pixel (${row}, ${col}) → ${j} → ${m} → output ${path.k} → the mistake.`}
          />
          <div className="pathmath">
            <div className="sub">This path&rsquo;s product of local slopes</div>
            <div className="formula mono">
              {factors.map(([label, v, digits], i) => (
                <span key={label} className="pf">
                  {i > 0 && ' × '}
                  <span className="pfv">({fmt(v, digits)})</span>
                  <span className="pfl">{label}</span>
                </span>
              ))}{' '}
              = <b>{signedSci(path.value)}</b> millionths
            </div>
          </div>
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
        These are part 2&rsquo;s &ldquo;no change&rdquo; connections.
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
            ? `Right: ${fmt(ch.gradient, 4)}. It’s positive, so the step lowers the weight of ${source} → ${wrong}, which lowers the wrong output.`
            : !checked.f
              ? `Not quite. Factor 3 is ${source}’s firing, ${fmt(ch.dzdw, 3)}, the same as for ${source} → ${LABEL}.`
              : `Factor 3 is right. Multiply all three: (${fmt(ch.dLdy, 3)})(${fmt(ch.dydz, 3)})(${fmt(ch.dzdw, 3)}) = ${fmt(ch.gradient, 4)}.`}
        </p>
      )}
    </form>
  )
}
