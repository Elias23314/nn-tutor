import { useMemo, useState, type ReactNode } from 'react'
import { Bridge, LayerHead, QuickCheck, RightIcon, SayItCard, WrongIcon } from '../components/LayerParts'
import { DEFAULT_SEED, IMAGE_SIDE, PIXELS, createNetwork, neuronName, predict } from '../engine/network'
import { fmt, thousands, type Connection } from '../playground/connection'
import { NetworkDiagram, type Mark } from '../playground/NetworkDiagram'
import { PixelImage } from '../playground/PixelImage'
import { useTrainer } from '../playground/useTrainer'
import { WeightCanvas } from '../playground/WeightPictures'
import { href } from '../routes'
import { correct, type Correction } from './correction'
import '../playground/playground.css'
import '../components/layer.css'
import './layer2.css'

const LABEL = 7
/** One round of practice, in plain words. Layer 3 adds "find the slope". */
const ROUND = ['Guess', 'Measure the mistake', 'Adjust', 'Repeat']

export function Layer2() {
  const { status } = useTrainer()
  return (
    <main className="layer">
      <LayerHead
        n={2}
        name="How it works"
        tag="No math"
        title="What changes inside it when it learns?"
        lede="So what is practice? One small routine, repeated thousands of times: show it a digit along with the right answer, see what it guesses, measure how wrong it was, and adjust. This page slows down a single round."
        loop={ROUND}
        loopCaption="One round of practice. The replay below walks through it."
      />
      <QuickCheck
        before="A neural network turns a picture into "
        after=", sends them along weighted connections through layers of neurons, and its biggest output is its guess."
        options={['rules', 'numbers', 'words']}
        answer="numbers"
        recap="A neural network turns a picture into numbers, sends them along weighted connections through layers of neurons, and its biggest output is its guess. At the end of part 1, practice turned a wrong guess into a right one."
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

type Guess = 'ink' | 'blank' | 'both'

/** The pixel → hidden-layer-1 connection this correction changes most: one concrete case to look at. */
function mostChanged(c: Correction): Connection {
  const g = c.grads.layers[0]
  let best = 0
  for (let k = 1; k < g.weights.length; k++) if (Math.abs(g.weights[k]) > Math.abs(g.weights[best])) best = k
  // Its pixel has ink by construction: connections from blank pixels have a gradient of 0.
  return { layer: 0, from: best % PIXELS, to: Math.floor(best / PIXELS) }
}

function Ready({ example }: { example: Float32Array }) {
  // The untrained network everyone starts from, and one correction on this 7.
  const c = useMemo(() => correct(createNetwork(undefined, DEFAULT_SEED), example, LABEL), [example])
  const one = useMemo(() => mostChanged(c), [c])
  const [step, setStep] = useState(1)
  const [guess, setGuess] = useState<Guess | null>(null)
  const [locked, setLocked] = useState(false)
  const [why, setWhy] = useState<number | null>(null)
  const [show, setShow] = useState<'before' | 'after'>('after')
  const adjusted = step === 3

  // What the diagram points at in each step.
  const marks: Mark[] =
    step === 1
      ? [{ layer: 'out', index: c.guessBefore }]
      : step === 2
        ? [
            { layer: 'out', index: c.guessBefore, label: `guess: ${fmt(c.before.yHat[c.guessBefore], 2)}\ngoal 0` },
            { layer: 'out', index: LABEL, label: `says ${fmt(c.before.yHat[LABEL], 2)}\ngoal 1` },
          ]
        : [{ layer: 1, index: one.to }]
  const caption =
    step === 1
      ? `The ${LABEL} flows through the untrained network, and its biggest output, ${c.guessBefore}, is its guess. Only the strongest connection into each neuron is drawn.`
      : step === 2
        ? `The right answer is ${LABEL}, so the ${LABEL} output should be 1 and every other output should be 0.`
        : 'Colored connections changed in this correction. Gray ones didn’t: nothing flowed through them, or the neuron they feed was silent. The outlined one is looked at more closely below.'

  return (
    <>
      <p className="body l2-note">
        Below we see a network whose input is a handwritten {LABEL}. This network hasn&rsquo;t practiced at all yet.
        Let&rsquo;s look at how the network can get better at correctly identifying this {LABEL} through
        &ldquo;training&rdquo;.
      </p>

      <Concepts />

      <div className="grid2 l2-round">
        <div className="l2-diagram" id="l2-diagram">
        <NetworkDiagram
          title="The network during this round"
          network={c.start}
          fwd={c.before}
          top={adjusted ? 2 : 1}
          changed={adjusted ? c.grads : null}
          unchangedLabel="didn’t change"
          changedDescription="This correction changes connections that had something flowing through them into a firing neuron, and leaves the rest alone."
          selected={adjusted ? one : null}
          marks={marks}
          caption={<span aria-live="polite">{caption}</span>}
        />
        </div>

        <div className="stack">
          <Replay c={c} step={step} setStep={setStep} locked={locked}>
            <Prediction
              guess={guess}
              setGuess={setGuess}
              locked={locked}
              lockIn={() => {
                setLocked(true)
                setStep(3)
              }}
            />
          </Replay>

          <div className="panel">
            <div className="ptitle" style={{ flexWrap: 'wrap' }}>
              <span className="h3">{step === 1 ? 'Its ten outputs' : step === 2 ? 'Each output against its goal' : 'Its outputs, before and after'}</span>
              {adjusted && (
                <div className="chips" role="group" aria-label="Which outputs to show">
                  <button className="chip" aria-pressed={show === 'before'} onClick={() => setShow('before')}>
                    Before
                  </button>
                  <button className="chip" aria-pressed={show === 'after'} onClick={() => setShow('after')}>
                    After
                  </button>
                </div>
              )}
            </div>
            <GoalBars yHat={adjusted && show === 'after' ? c.after.yHat : c.before.yHat} goals={step >= 2} />
            {step >= 2 && (
              <p className="body" style={{ marginTop: 12 }} aria-live="polite">
                {adjusted && show === 'after' ? (
                  <>
                    After the correction it says {c.guessAfter}, and the mistake is{' '}
                    <span className="mono">{fmt(c.lossAfter)}</span>, down from <span className="mono">{fmt(c.lossBefore)}</span>.
                    Every output moved a little toward its goal.
                  </>
                ) : (
                  <>
                    The blue tick is each output&rsquo;s goal, and the gap is how far the output is from it. The cost
                    function combines all ten gaps into one number, the mistake:{' '}
                    <span className="mono">{fmt(c.lossBefore)}</span>.
                  </>
                )}
              </p>
            )}
          </div>
        </div>
      </div>

      {adjusted && (
        <div className="panel sec">
          <div className="h3" style={{ marginBottom: 12 }}>
            A closer look at what changed
          </div>

          <p className="body" aria-live="polite">
            {guess === 'ink' ? (
              <>
                You predicted it: only connections from the {c.inkPixels} pixels with ink changed. Not one connection from a
                blank pixel did.
              </>
            ) : (
              <>
                {guess === 'blank'
                  ? 'Not quite: it’s the opposite. Only '
                  : 'Not quite. You predicted connections from both kinds of pixel would change, but only '}
                connections from the {c.inkPixels} pixels with ink changed. Not one connection from a blank pixel did. The
                question below gets at why.
              </>
            )}
          </p>

          <WhyBlank why={why} setWhy={setWhy} changed={c.changed} />

          {why !== null && (
            <>
              <p className="callout" style={{ marginTop: 16 }}>
                The rule: a connection changes only if it played a part in the mistake. That means something flowed
                through it, and the neuron it feeds was firing. A blank pixel sends 0, and a silent neuron passes nothing
                on, so either way that connection is left alone.
              </p>

              <OneConnection c={c} one={one} />

              <div className="sub" style={{ margin: '20px 0 8px' }}>
                Where the changes landed: the connections from the pixels into hidden layer 1
              </div>
              <div className="seen-layout">
                <div>
                  <PixelImage image={example} size={140} label={`The ${LABEL} it saw`} />
                  <p className="small" style={{ marginTop: 8 }}>
                    The {LABEL} it saw
                  </p>
                </div>
                <div>
                  <ChangeGrid c={c} />
                  <p className="body" style={{ marginTop: 12 }}>
                    Each square is one neuron in hidden layer 1, and shows how its 784 connections from the pixels
                    changed: blue ones now push it up more, orange ones push it down more. The changes trace the{' '}
                    {LABEL}&rsquo;s ink, and the {16 - c.firing1} silent neurons didn&rsquo;t change at all. That is{' '}
                    {c.inkPixels} ink pixels × {c.firing1} firing neurons = {thousands(c.changedFromPixels)} connections.
                    The other {thousands(c.changed - c.changedFromPixels)} changes are deeper in the network.
                  </p>
                </div>
              </div>

              <p className="body l2-scale">
                <b>One correction is only a nudge</b>, and a single example can&rsquo;t teach it every digit. Real practice
                repeats this round thousands of times. To go faster, each round looks at 32 digits together and makes one
                correction that helps on all of them on average; {thousands(1875)} such rounds covers all 60,000 examples
                once. That is what turned part 1&rsquo;s &ldquo;before practice&rdquo; into its &ldquo;after
                practice&rdquo;.
              </p>
            </>
          )}
        </div>
      )}

      <div className="sec">
        <SayItCard
          before="Learning means changing "
          answer="weights"
          after=": after a mistake, only the connections that played a part in it change."
          accepts={(t) => /connection|weight|strength|synap/i.test(t)}
          next={{
            label: 'Part 3: With numbers',
            text: 'So a correction changes weights, but by how much, and in which direction? Part 3 zooms in on one connection. You’ll nudge it by hand, measure how the mistake responds, and work out the step it should take.',
          }}
        />
      </div>

      <div className="exits">
        <a className="btn sec" href={href('layer-1')}>
          Back to part 1
        </a>
        <a className="btn" href={href('layer-3')}>
          Go deeper: part 3
        </a>
      </div>
    </>
  )
}

/** The three ideas a round of practice rests on, before the replay shows them on one 7. */
function Concepts() {
  return (
    <div className="l2-concepts">
      <div className="panel">
        <div className="h3" style={{ marginBottom: 10 }}>
          <span className="fnum-lg">1</span> Labeled data: examples with the answer attached
        </div>
        <p className="body">
          To practice, the network needs examples where the right answer is already known. Each picture comes with a{' '}
          <b>label</b>: the digit a person said it is. This {LABEL} is one of 60,000 labeled digits. The network
          doesn&rsquo;t see the label when it guesses; the label is used afterwards to check the guess.
        </p>
        <Bridge>Once it has guessed, the label tells us how wrong it was. That&rsquo;s step 2.</Bridge>
      </div>

      <div className="panel">
        <div className="h3" style={{ marginBottom: 10 }}>
          <span className="fnum-lg">2</span> The cost function: how wrong was it?
        </div>
        <p className="body">
          The <b>cost function</b> compares two things: what the network guessed (its ten scores) and what it should have
          said (1 for the labeled digit, 0 for the other nine). It turns the difference into one number, the{' '}
          <b>cost</b>. We&rsquo;ll also call it the <b>mistake</b>. Far off: a big cost. Exactly right: 0. Practice is all
          about making that number smaller.
        </p>
        <Bridge>The cost says how wrong it was. What it does about it is step 3.</Bridge>
      </div>

      <div className="panel">
        <div className="h3" style={{ marginBottom: 10 }}>
          <span className="fnum-lg">3</span> Changing the weights to lower the cost
        </div>
        <p className="body">
          The only things the network can change are its weights, plus each neuron&rsquo;s <b>bias</b> (the starting number
          a neuron adds to its total, from part 1). After each guess, every weight that helped
          cause the mistake is nudged a little, in whichever direction would make the cost smaller. One nudge barely
          changes anything; thousands of them, over thousands of labeled digits, is how its guesses get better.
        </p>
        <Bridge>Now watch one full round of this on the {LABEL} below.</Bridge>
      </div>
    </div>
  )
}

function Replay({
  c,
  step,
  setStep,
  locked,
  children,
}: {
  c: Correction
  step: number
  setStep: (s: number) => void
  locked: boolean
  /** The prediction, asked between measuring and adjusting. */
  children: ReactNode
}) {
  const top = c.before.yHat[c.guessBefore]
  const wrong = c.guessBefore !== LABEL
  const steps = [
    {
      title: 'Guess',
      text: (
        <>
          It&rsquo;s shown the {LABEL}, but not told the answer. The 784 numbers flow through its layers, just as in part
          1, and its biggest output is {c.guessBefore} (<span className="mono">{fmt(top, 2)}</span>). Its weights are
          still random, so {wrong ? 'the guess is wrong' : 'a right guess is only luck'}.
        </>
      ),
    },
    {
      title: 'Measure the mistake',
      text: (
        <>
          Now the right answer is revealed: {LABEL}. That sets a goal for every output: 1 for the {LABEL}, and 0 for the
          other nine. The cost function combines how far all ten outputs are from their goals into one number, the{' '}
          <b>mistake</b>. Big gaps count extra, so it isn&rsquo;t a simple total. 0 would be perfect. Here it is{' '}
          <span className="mono">{fmt(c.lossBefore)}</span>.
          <span className="small" style={{ display: 'block', marginTop: 6 }}>
            (It&rsquo;s the average of the ten gaps, each squared. Part 3 uses that.)
          </span>
        </>
      ),
    },
    {
      title: 'Adjust',
      text: (
        <>
          Each weight that played a part in the mistake is changed a little, in the direction that would have
          made the mistake smaller. {thousands(c.changed)} of the network&rsquo;s {thousands(c.total)} adjustable numbers
          changed. (Those {thousands(c.total)} are about 13,000 weights across all its layers, plus one bias for each of its
          42 neurons.) Afterwards it says{' '}
          {c.guessAfter} for this{' '}
          {LABEL}. This adjustment is called a <b>correction</b>. But it&rsquo;s only one nudge: the {LABEL} output rose
          from <span className="mono">{fmt(c.before.yHat[LABEL], 2)}</span> to{' '}
          <span className="mono">{fmt(c.after.yHat[LABEL], 2)}</span>, still far from its goal of 1.
          {c.guessAfter === LABEL ? ' Here it was just enough to tip the guess.' : ''}
        </>
      ),
    },
  ]
  return (
    <div className="panel">
      <div className="ptitle">
        <span className="h3">Replay one round</span>
        <span className="small">Part {step} of 3</span>
      </div>
      <ol className="steps" style={{ listStyle: 'none' }}>
        {steps.map((s, i) => {
          const n = i + 1
          const cls = n < step ? 'done' : n === step ? 'cur' : 'todo'
          return (
            <li className={`st ${cls}`} key={s.title} aria-current={n === step ? 'step' : undefined}>
              <span className="n">{n}</span>
              <div>
                <div className="tt">{s.title}</div>
                {n <= step && <div className="tx">{s.text}</div>}
                {n === step && (
                  <button
                    className="lnk l2-see"
                    onClick={() => document.getElementById('l2-diagram')?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
                  >
                    See it in the diagram ↓
                  </button>
                )}
                {n === 2 && step >= 2 && children}
              </div>
            </li>
          )
        })}
      </ol>
      <div className="btns" style={{ marginTop: 16 }}>
        <button className="btn sec sm" disabled={step === 1} onClick={() => setStep(step - 1)}>
          Back
        </button>
        {step === 1 && (
          <button className="btn sm" onClick={() => setStep(2)}>
            Next: measure the mistake
          </button>
        )}
        {step === 2 && locked && (
          <button className="btn sm" onClick={() => setStep(3)}>
            Next: adjust
          </button>
        )}
        {step === 2 && !locked && <span className="small">Make your prediction above to continue.</span>}
        {step === 3 && (
          <button className="btn sm" onClick={() => setStep(1)}>
            Replay
          </button>
        )}
      </div>
    </div>
  )
}

const GUESSES: [Guess, string][] = [
  ['ink', 'Only from pixels with ink'],
  ['blank', 'Only from blank pixels'],
  ['both', 'From both'],
]

function Prediction({
  guess,
  setGuess,
  locked,
  lockIn,
}: {
  guess: Guess | null
  setGuess: (g: Guess) => void
  locked: boolean
  lockIn: () => void
}) {
  return (
    <div className="l2-predict">
      <div className="pk" id="pick-q">
        {locked ? 'Your prediction' : 'Predict before it adjusts'}
      </div>
      <p className="body">
        Every pixel is connected to hidden layer 1. Which of those connections do you think will change?
        {!locked && ' Think back to part 1: what does a blank pixel send?'}
      </p>
      <div className="chips" role="group" aria-labelledby="pick-q">
        {GUESSES.map(([g, label]) => (
          <button key={g} className="chip" aria-pressed={guess === g} disabled={locked} onClick={() => setGuess(g)}>
            {label}
            {locked && guess === g && (g === 'ink' ? <RightIcon /> : <WrongIcon />)}
          </button>
        ))}
      </div>
      {!locked && (
        <button className="btn sm" style={{ alignSelf: 'flex-start' }} disabled={guess === null} onClick={lockIn}>
          Lock in and adjust
        </button>
      )}
    </div>
  )
}

/** The ten outputs as bars, like layer 1's, with each one's goal and gap once the answer is known. */
function GoalBars({ yHat, goals }: { yHat: ArrayLike<number>; goals: boolean }) {
  const guess = predict(yHat)
  return (
    <div className="gbars" role="list" aria-label="Its ten outputs">
      {Array.from({ length: 10 }, (_, d) => {
        const v = yHat[d]
        const goal = d === LABEL ? 1 : 0
        return (
          <div className={`grow${d === guess ? ' top' : ''}`} key={d} role="listitem">
            <span className="dg">{d}</span>
            <div className="tr">
              <i style={{ width: `${(v * 100).toFixed(1)}%` }} />
              {goals && <span className="goal" style={{ left: `${goal * 100}%` }} />}
            </div>
            <span className="vv">{fmt(v, 2)}</span>
            <span className="nt">
              {goals ? `goal ${goal} · gap ${fmt(Math.abs(v - goal), 2)}` : d === guess ? 'its guess' : ''}
            </span>
          </div>
        )
      })}
    </div>
  )
}

const WHY = [
  'A blank pixel sent 0, so its connections played no part in the mistake',
  'Those connections were already right',
  'It can only change one connection at a time',
]

function WhyBlank({ why, setWhy, changed }: { why: number | null; setWhy: (i: number) => void; changed: number }) {
  const feedback = [
    'Yes. A connection’s change depends on what flowed through it, and nothing flowed from a blank pixel.',
    'Not quite. Their weights are as random as every other connection’s. They didn’t change because nothing flowed through them: a blank pixel sends 0.',
    `Not quite. This one correction changed ${thousands(changed)} numbers at once. Blank pixels’ connections were left out because nothing flowed through them.`,
  ]
  return (
    <div className="l2-why">
      <div className="q" style={{ marginBottom: 12, fontSize: 18 }} id="why-q">
        Why do you think the connections from blank pixels didn&rsquo;t change?
      </div>
      <div className="opts" role="radiogroup" aria-labelledby="why-q">
        {WHY.map((label, i) => (
          <button
            key={label}
            role="radio"
            aria-checked={why === i}
            className={`opt${why === i ? ' mark' : ''}`}
            style={{ background: 'var(--card)', textAlign: 'left' }}
            disabled={why === 0}
            onClick={() => setWhy(i)}
          >
            {why === i ? i === 0 ? <RightIcon /> : <WrongIcon /> : <span className="r" />}
            {label}
          </button>
        ))}
      </div>
      {why !== null && (
        <p className="feedback" aria-live="polite">
          {feedback[why]}
        </p>
      )}
    </div>
  )
}

/** One connection's strength before and after: the concrete case behind "connection strengths change". */
function OneConnection({ c, one }: { c: Correction; one: Connection }) {
  const at = one.to * PIXELS + one.from
  const before = c.start.layers[0].weights[at]
  const after = c.network.layers[0].weights[at]
  const name = neuronName(1, one.to)
  const row = Math.floor(one.from / IMAGE_SIDE) + 1
  const col = (one.from % IMAGE_SIDE) + 1
  return (
    <div className="l2-one">
      <div className="sub">One connection, before and after</div>
      <div className="l2-one-row">
        <div className="fbox2">
          <span className="fl">before</span>
          <span className="fv mono">{fmt(before)}</span>
        </div>
        <span className="fop" aria-hidden="true">
          →
        </span>
        <div className="fbox2 changed">
          <span className="fl">after</span>
          <span className="fv mono">{fmt(after)}</span>
        </div>
        <p className="body">
          This is the weight of the connection from the pixel at row {row}, column {col} to neuron {name} in hidden layer
          1: the outlined line in the diagram. It went {after > before ? 'up' : 'down'} by{' '}
          <span className="mono">{fmt(Math.abs(after - before))}</span>, so that pixel now pushes neuron {name}{' '}
          {after > before ? (after > 0 ? 'up a little more' : 'down a little less') : after < 0 ? 'down a little more' : 'up a little less'}.
          That small change to a number is what learning
          physically is. This correction made {thousands(c.changed)} of them.
        </p>
      </div>
    </div>
  )
}

/** How each hidden-layer-1 neuron's 784 connections changed in this step (−η · gradient). */
function ChangeGrid({ c }: { c: Correction }) {
  const g = c.grads.layers[0]
  const rows = useMemo(
    () =>
      Array.from({ length: g.outSize }, (_, j) =>
        Float64Array.from(g.weights.subarray(j * PIXELS, (j + 1) * PIXELS), (v) => (v === 0 ? 0 : -v)),
      ),
    [g],
  )
  let max = 1e-12
  for (const r of rows) for (const v of r) max = Math.max(max, Math.abs(v))
  return (
    <div className="wgrid">
      {rows.map((r, j) => {
        const silent = c.before.layers[0].a[j] === 0
        return (
          <div key={j}>
            <WeightCanvas weights={r} max={max} name={`${neuronName(1, j)}${silent ? ' (silent, unchanged)' : ''}`} />
            <div className="cap">
              {neuronName(1, j)}
              {silent ? ' · silent' : ''}
            </div>
          </div>
        )
      })}
    </div>
  )
}
