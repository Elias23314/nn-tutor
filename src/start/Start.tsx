import { useMemo, useState } from 'react'
import { forward, predict } from '../engine/network'
import { fmt, thousands } from '../playground/connection'
import { NetworkDiagram } from '../playground/NetworkDiagram'
import type { Snapshot } from '../playground/protocol'
import { useTrainer } from '../playground/useTrainer'
import { href, type Route } from '../routes'
import '../playground/playground.css'
import '../components/layer.css'
import './start.css'

/** One pass through the 60,000 training digits (60,000 / 32). */
const ONE_PASS = 1_875

interface Rung {
  n: string
  question: string
  name: string
  tag: string
  blurb: string
  route: Route | null
}

const RUNGS: Rung[] = [
  {
    n: '1',
    question: 'Does it actually get better with practice?',
    name: 'What it does',
    tag: 'No math',
    blurb: 'Watch it practice on handwritten digits, and predict which digit it masters last.',
    route: 'layer-1',
  },
  {
    n: '2',
    question: 'What physically changes when it learns?',
    name: 'How it works',
    tag: 'No math',
    blurb: 'Slow one correction down and see which of its 13,002 numbers change.',
    route: 'layer-2',
  },
  {
    n: '3',
    question: 'How much should one connection change?',
    name: 'With numbers',
    tag: 'Arithmetic',
    blurb: 'Nudge one connection, measure the mistake, and take a step.',
    route: null,
  },
  {
    n: '4',
    question: 'How does it get that number without nudging?',
    name: 'The math',
    tag: 'Calc I',
    blurb: 'Follow the chain rule backward, then see how deeper connections add up many paths.',
    route: null,
  },
  {
    n: 'P',
    question: 'What will it do, and can I break it?',
    name: 'Playground',
    tag: 'Any depth',
    blurb: 'Train it, draw your own digits, edit its examples, and try to make it fail.',
    route: 'playground',
  },
]

const QUESTIONS = [
  {
    q: 'How can a few stickers fool a stop-sign detector?',
    a: 'A network learns whatever shrinks its mistakes on its examples. That isn’t always the idea we meant.',
    where: 'Answered in layer 3 and the playground',
  },
  {
    q: 'Is a neural network really like a brain?',
    a: 'Parts of the comparison hold up. Each layer shows exactly where it holds and where it breaks.',
    where: 'Answered across layers 2, 3 and 4',
  },
  {
    q: 'Why does it matter which examples it learns from?',
    a: 'It has only ever seen 60,000 handwritten digits. Change them, and it learns something else.',
    where: 'Answered in layer 1 and the playground',
  },
]

export function Start() {
  const [calculus, setCalculus] = useState<'no' | 'yes' | null>(null)
  // Layer 2 is the deepest finished layer; once layers 3 and 4 exist, "yes" should point at layer 3.
  const recommended: Route = calculus === 'yes' ? 'layer-2' : 'layer-1'

  return (
    <main className="layer start">
      <div className="hero">
        <div>
          <h1>How does a machine learn?</h1>
          <p className="lede">
            Nobody writes its rules. Watch a network learn to read handwritten digits, one layer at a time, from no math
            to the chain rule.
          </p>
          <div className="btns" style={{ marginTop: 32 }}>
            <a className="btn big" href={href('layer-1')}>
              Start with layer 1 (no math)
            </a>
            <a className="lnk" href={href('playground')}>
              Skip to the playground
            </a>
          </div>
          <p className="small" style={{ marginTop: 14 }}>
            The playground works best after layer 1.
          </p>
        </div>
        <HeroNetwork />
      </div>

      <h2>Three questions it helps you answer</h2>
      <div className="grid3">
        {QUESTIONS.map((x) => (
          <div className="panel question" key={x.q}>
            <div className="h3">{x.q}</div>
            <p className="body">{x.a}</p>
            <span className="small where">{x.where}</span>
          </div>
        ))}
      </div>

      <div className="choose">
        <div>
          <h2>Choose how deep to go</h2>
          <p className="body intro">
            Every layer tells the same story about the same network, in more detail. Each one is complete on its own, so
            you can stop whenever you like.
          </p>
          <ol className="rungs">
            {RUNGS.map((r) => {
              const isRecommended = r.route === recommended
              return (
                <li className={`rungcard${isRecommended ? ' rec' : ''}${r.route ? '' : ' soon'}`} key={r.n}>
                  <div className="num">{r.n}</div>
                  <div className="body-col">
                    <div className="row">
                      <span className="h3">{r.question}</span>
                      {isRecommended && <span className="tag dark">{calculus === 'yes' ? 'Suggested for you' : 'Recommended start'}</span>}
                    </div>
                    <div className="row">
                      <span className="name">{r.name}</span>
                      <span className="tag">{r.tag}</span>
                    </div>
                    <p className="small">{r.blurb}</p>
                  </div>
                  {r.route ? (
                    <a className={`btn${isRecommended ? '' : ' sec sm'}`} href={href(r.route)}>
                      {isRecommended ? `Start ${r.n === 'P' ? 'the playground' : `layer ${r.n}`}` : 'Open'}
                    </a>
                  ) : (
                    <span className="small soon-note">Coming soon</span>
                  )}
                </li>
              )
            })}
          </ol>
        </div>

        <div className="panel unsure">
          <div className="h3" id="calc-q">
            Not sure where to start?
          </div>
          <p className="body">Have you taken calculus?</p>
          <div className="chips" role="group" aria-labelledby="calc-q">
            <button className="chip" aria-pressed={calculus === 'no'} onClick={() => setCalculus('no')}>
              Not yet
            </button>
            <button className="chip" aria-pressed={calculus === 'yes'} onClick={() => setCalculus('yes')}>
              Yes
            </button>
          </div>
          <p className="small" style={{ fontSize: 15 }} aria-live="polite">
            {calculus === 'yes'
              ? 'Layers 3 and 4 use arithmetic and Calc I, and they’re still being built. Until then, layer 2 is the deepest finished layer. Starting at layer 1 still gives you the whole story.'
              : calculus === 'no'
                ? 'We’ll suggest layer 1. No math needed: you’ll watch it practice and make predictions. You can jump to any layer from the ladder at the top of every page.'
                : 'Pick one and we’ll suggest a layer. You can jump to any layer from the ladder at the top of every page.'}
          </p>
        </div>
      </div>
    </main>
  )
}

/** The network on the page, practicing one pass from scratch, then reading a 7. */
function HeroNetwork() {
  // Score the test digits only at the end: the hero just needs the network to practice.
  const { status } = useTrainer([
    { type: 'setTestEvery', rounds: ONE_PASS },
    { type: 'setSpeed', speed: 'fast' },
    { type: 'practice', rounds: ONE_PASS },
  ])
  return (
    <div className="hero-net">
      {status.kind === 'ready' ? (
        <HeroDiagram state={status.state} seven={status.samples[7]} />
      ) : (
        <div className="panel hero-wait">
          {status.kind === 'error' ? `Couldn’t start the network: ${status.message}` : 'Loading 70,000 handwritten digits…'}
        </div>
      )}
    </div>
  )
}

function HeroDiagram({ state, seven }: { state: Snapshot; seven: Float32Array }) {
  const fwd = useMemo(() => forward(state.network, seven), [state.network, seven])
  const guess = predict(fwd.yHat)
  const done = state.pending === 0 && state.round >= ONE_PASS
  return (
    <NetworkDiagram
      title={null}
      network={state.network}
      fwd={fwd}
      caption={
        <>
          The same network appears on every page: 784 pixels in, two layers of 16 neurons, ten outputs.{' '}
          <span aria-live="polite">
            {done
              ? `Here it reads a 7 after practice: it says ${guess} (${fmt(fwd.yHat[guess], 2)}).`
              : `It’s practicing right now, round ${thousands(state.round)} of ${thousands(ONE_PASS)}. For this 7 it says ${guess} (${fmt(fwd.yHat[guess], 2)}).`}
          </span>
        </>
      }
    />
  )
}
