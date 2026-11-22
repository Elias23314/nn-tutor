import { useState } from 'react'
import { LayerHead, SayItCard } from '../components/LayerParts'
import type { Snapshot } from '../playground/protocol'
import { useTrainer, type Send } from '../playground/useTrainer'
import { href } from '../routes'
import { WhatIsIt } from './WhatIsIt'
import '../playground/playground.css'
import '../components/layer.css'

/** The digit the page follows from picture to guess. */
const INTRO_DIGIT = 7
/** One pass through the 60,000 training digits (60,000 / 32): what "after practice" means here. */
const ONE_PASS = 1_875

// The practice section (prediction, ten digits, practice chart, reveal, mistake panel, weight pictures and
// brain table) was taken out to keep layer 1 simple. It's in git history (2e2c631) if it comes back;
// story.ts and PracticeChart.tsx are kept for that.
export function Layer1() {
  // Score the test digits only at the end: this page just needs the network before and after practice.
  const { status, send } = useTrainer([
    { type: 'setTestEvery', rounds: ONE_PASS },
    { type: 'setSpeed', speed: 'fast' },
  ])
  return (
    <main className="layer">
      <LayerHead
        n={1}
        name="What it does"
        tag="No math"
        title="What is a neural network?"
        lede="This one has a single job: look at a handwritten digit and say which digit it is. Follow one 7 from picture to guess."
      />
      {status.kind === 'loading' && <p className="notice">Loading 70,000 handwritten digits: 60,000 for it to practice on, and 10,000 kept aside to test it on digits it has never seen…</p>}
      {status.kind === 'error' && (
        <p className="notice" role="alert">
          Couldn&rsquo;t start the network: {status.message}
        </p>
      )}
      {status.kind === 'ready' && <Ready state={status.state} image={status.samples[INTRO_DIGIT]} send={send} />}
    </main>
  )
}

function Ready({ state, image, send }: { state: Snapshot; image: Float32Array; send: Send }) {
  const [view, setView] = useState<'before' | 'after'>('before')
  return (
    <>
      <WhatIsIt
        image={image}
        network={view === 'after' ? state.network : state.initialNetwork}
        digit={INTRO_DIGIT}
        practice={{
          view,
          round: Math.min(state.round, ONE_PASS),
          total: ONE_PASS,
          setView: (v) => {
            setView(v)
            // The first switch to "after" runs one pass of practice; later switches just show the result.
            if (v === 'after' && state.round === 0 && state.pending === 0) send({ type: 'practice', rounds: ONE_PASS })
          },
        }}
      />

      <div className="sec">
        <SayItCard
          before="A neural network turns a picture into "
          answer="numbers"
          after=", sends them along weighted connections through layers of neurons, and its biggest output is its guess."
          accepts={(t) => /number|value|input|pixel/i.test(t)}
          next={{
            label: 'Part 2: How it works',
            text: 'Practice changed its guess from wrong to right, and nobody wrote the right weights in. So what does practice actually do? Part 2 slows down a single round of it, so you can see exactly what changes inside the network when it gets something wrong.',
          }}
        />
      </div>

      <div className="exits">
        <a className="btn sec" href={href('start')}>
          Back to start here
        </a>
        <a className="btn" href={href('layer-2')}>
          Go deeper: part 2
        </a>
      </div>
    </>
  )
}
