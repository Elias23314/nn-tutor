import { LayerHead, SayItCard } from '../components/LayerParts'
import { useTrainer } from '../playground/useTrainer'
import { href } from '../routes'
import { WhatIsIt } from './WhatIsIt'
import '../playground/playground.css'
import '../components/layer.css'

/** The digit the opening uses to show pixels becoming numbers. */
const INTRO_DIGIT = 7

// The practice section (prediction, ten digits, practice chart, reveal, mistake panel, weight pictures and
// brain table) was taken out to keep layer 1 simple. It's in git history (2e2c631) if it comes back;
// story.ts and PracticeChart.tsx are kept for that.
export function Layer1() {
  const { status } = useTrainer()
  return (
    <main className="layer">
      <LayerHead
        n={1}
        name="What it does"
        tag="No math"
        title="Does it actually get better with practice?"
        lede="Its job: name the digit in a 28 × 28 image. Nobody tells it what a 7 looks like. It only gets corrected on examples."
        loopCaption="Each round of practice runs this whole loop once"
      />
      {status.kind === 'loading' && <p className="notice">Loading 70,000 handwritten digits…</p>}
      {status.kind === 'error' && (
        <p className="notice" role="alert">
          Couldn&rsquo;t start the network: {status.message}
        </p>
      )}
      {status.kind === 'ready' && (
        <>
          <WhatIsIt image={status.samples[INTRO_DIGIT]} network={status.state.network} digit={INTRO_DIGIT} />

          <div className="sec">
            <SayItCard
              before="Nobody told it what a 7 looks like. It got better by being "
              answer="corrected"
              after=" on examples."
              accepts={(t) => /correct|feedback|fix|told.*wrong|mistake/i.test(t)}
            />
          </div>

          <div className="exits">
            <a className="btn sec" href={href('start')}>
              Back to start here
            </a>
            <a className="btn" href={href('layer-2')}>
              Go deeper: layer 2
            </a>
          </div>
        </>
      )}
    </main>
  )
}
