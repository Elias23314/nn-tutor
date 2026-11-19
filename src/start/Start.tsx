import { useMemo } from 'react'
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
    question: 'What is a neural network?',
    name: 'What it does',
    tag: 'No math',
    blurb: 'Start here. Follow a handwritten 7 as it becomes 784 numbers, flows through layers of neurons, and comes out as a guess.',
    route: 'layer-1',
  },
  {
    n: '2',
    question: 'What changes inside it when it learns?',
    name: 'How it works',
    tag: 'No math',
    blurb: 'Part 1 shows practice turning a wrong guess into a right one. This slows down one round of practice to show what changes inside: its weights.',
    route: 'layer-2',
  },
  {
    n: '3',
    question: 'How much should one connection change?',
    name: 'With numbers',
    tag: 'Arithmetic',
    blurb: 'Part 2 shows that connections change a little. This works out, with plain arithmetic, which way and how far one connection should change.',
    route: 'layer-3',
  },
  {
    n: '4',
    question: 'How does it get that number without nudging?',
    name: 'The math',
    tag: 'Calculus',
    blurb: 'Part 3 works out one connection by trial and error. This shows the shortcut that works out every connection at once.',
    route: 'layer-4',
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
    q: 'Why can AI (neural networks) confuse two things that are drastically different?',
    a: 'A neural network is shown thousands of examples with the right answer attached, and it adjusts itself until it gets most of them right. It only learns whatever patterns help it do that. It has no common sense the way people do, so if two very different things share similar shapes or patterns, it can mix them up.',
    where: 'Parts 2 and 3, which show how it shrinks its mistakes',
  },
  {
    q: 'Why is the data we train a neural network on important?',
    a: 'The data is where the network learns its patterns. Bad data teaches it the wrong patterns. Biased data does too: a network trained to recognize faces using only photos of white men learns to recognize white men’s faces, but struggles to generalize to everyone else.',
    where: 'The playground, where you can remove or relabel its examples',
  },
  {
    q: 'How can a machine learn when nobody tells it the rules?',
    a: 'Nobody writes rules for what a 7 looks like. The network starts with random weights, guesses, and is told how wrong it was. Each mistake nudges its weights a little in the direction that would have made the mistake smaller. Repeat that thousands of times and the patterns emerge on their own.',
    where: 'Parts 2 and 3, step by step',
  },
]

export function Start() {
  // Parts 1 and 2 define the words parts 3 and 4 use, so everyone starts at part 1.
  const recommended: Route = 'layer-1'

  return (
    <main className="layer start">
      <div className="hero">
        <div>
          <h1>How does a machine learn?</h1>
          <p className="lede">
            Let&rsquo;s dive into one of the core components of AI literacy: how machines learn. By the end of this guided
            experience, you should be able to describe how a neural network learns to recognize things when people give it
            examples with the right answers attached, but nobody writes the rules.
          </p>
          <div className="btns" style={{ marginTop: 32 }}>
            <a className="btn big" href={href('layer-1')}>
              Start with part 1 (no math)
            </a>
            <a className="lnk" href={href('playground')}>
              Skip to the playground
            </a>
          </div>
          <p className="small" style={{ marginTop: 14 }}>
            The playground works best after part 1.
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
            <p className="small where">
              <span className="k">Covered in</span> {x.where}
            </p>
          </div>
        ))}
      </div>

      <div className="choose">
        <div>
          <h2>Choose how deep to go</h2>
          <p className="body intro">
            Every part follows the same network reading the same handwritten 7. Each one picks up where the last
            stopped and goes one level deeper, so take them in order. You can stop after any part and still have a
            complete picture at that depth.
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
                      {isRecommended && <span className="tag dark">Recommended start</span>}
                    </div>
                    <div className="row">
                      <span className="name">{r.name}</span>
                      <span className="tag">{r.tag}</span>
                    </div>
                    <p className="small">{r.blurb}</p>
                  </div>
                  {r.route ? (
                    <a className={`btn${isRecommended ? '' : ' sec sm'}`} href={href(r.route)}>
                      {isRecommended ? `Start ${r.n === 'P' ? 'the playground' : `part ${r.n}`}` : 'Open'}
                    </a>
                  ) : (
                    <span className="small soon-note">Coming soon</span>
                  )}
                </li>
              )
            })}
          </ol>
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
          {status.kind === 'error' ? `Couldn’t start the network: ${status.message}` : 'Loading 70,000 handwritten digits: 60,000 for it to practice on, and 10,000 kept aside to test it on digits it has never seen…'}
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
      legend={false}
      network={state.network}
      fwd={fwd}
      caption={
        <>
          A picture goes in on the left as 784 pixels and ten scores come out on the right, one for each digit 0&ndash;9.
          The biggest score is its guess. Part 1 explains every part.{' '}
          <span aria-live="polite">
            {done
              ? `Here it reads a 7 after practice: its biggest score is ${guess} (${fmt(fwd.yHat[guess], 2)} out of 1).`
              : `It’s practicing right now, round ${thousands(state.round)} of ${thousands(ONE_PASS)}. For this 7 its biggest score is ${guess} (${fmt(fwd.yHat[guess], 2)} out of 1).`}
          </span>
        </>
      }
    />
  )
}
