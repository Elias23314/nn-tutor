// Trains the network off the main thread and reports its state to the page.
import { getSample, loadMnist } from '../engine/mnist'
import { forward, predict } from '../engine/network'
import { Trainer } from '../engine/trainer'
import type { Command, ExperimentResult, Report, Snapshot, Speed } from './protocol'

const post = (report: Report) => self.postMessage(report)

/**
 * How long to train between reports, and how often to score the 10,000 test digits.
 * Scoring costs about as much as 100 rounds of practice, so it's frequent only early on,
 * where the accuracy curve changes fastest.
 */
const PACE: Record<Speed, { sliceMs: number; restMs: number; testEvery: (round: number) => number }> = {
  slow: { sliceMs: 0, restMs: 120, testEvery: () => 5 },
  normal: { sliceMs: 60, restMs: 0, testEvery: (r) => (r < 100 ? 10 : r < 400 ? 25 : 100) },
  fast: { sliceMs: 250, restMs: 0, testEvery: (r) => (r < 400 ? 50 : 250) },
}

let trainer: Trainer | null = null
let playing = false
let pending = 0
let speed: Speed = 'normal'
let running = false
let lastTestRound = 0
let testEvery: number | null = null
const queue: Command[] = []
/** The experiment being run, and the results of recent ones that finished. */
let experiment: { id: number; probe?: Float32Array } | null = null
const experiments: ExperimentResult[] = []
const KEEP = 10

function finishExperiment(t: Trainer) {
  if (!experiment) return
  const probe = experiment.probe ? forward(t.network, experiment.probe).yHat : null
  const digit = probe ? predict(probe) : 0
  experiments.push({
    id: experiment.id,
    accuracy: t.lastTest?.accuracy ?? 0,
    collapsed: t.collapse() !== null,
    probe: probe ? { digit, value: probe[digit] } : null,
  })
  if (experiments.length > KEEP) experiments.shift()
  experiment = null
}

function snapshot(t: Trainer): Snapshot {
  return {
    session: t.session,
    experiments,
    round: t.round,
    roundsPerPass: t.roundsPerPass,
    eta: t.eta,
    speed,
    playing,
    pending,
    recentLoss: t.recentLoss,
    lossHistory: t.lossHistory(),
    accuracyHistory: t.accuracyHistory,
    digitHistory: t.digitHistory,
    lastTest: t.lastTest,
    collapse: t.collapse(),
    labelCounts: t.labelCounts(),
    trainCount: t.train.count,
    changes: t.changes,
    network: t.network,
    initialNetwork: t.initialNetwork,
  }
}

function report(t: Trainer) {
  post({ type: 'state', state: snapshot(t) })
}

function testNow(t: Trainer) {
  t.test()
  lastTestRound = t.round
}

/** Trains in short slices so commands (like Pause) are handled between them. */
async function run(t: Trainer) {
  if (running) return
  running = true
  while (playing || pending > 0) {
    const pace = PACE[speed]
    const start = performance.now()
    do {
      t.practiceOnce()
      if (pending > 0) pending--
      if (t.round - lastTestRound >= (testEvery ?? pace.testEvery(t.round))) testNow(t)
    } while ((playing || pending > 0) && performance.now() - start < pace.sliceMs)
    if (!playing && pending === 0 && lastTestRound !== t.round) testNow(t)
    if (pending === 0) finishExperiment(t)
    report(t)
    await new Promise((resolve) => setTimeout(resolve, pace.restMs))
  }
  running = false
}

function handle(t: Trainer, cmd: Command) {
  // Stopping, restarting or changing the examples interrupts a running experiment.
  if (['pause', 'reset', 'setExamples', 'play', 'practice'].includes(cmd.type)) experiment = null
  switch (cmd.type) {
    case 'practice':
      pending += cmd.rounds
      break
    case 'play':
      playing = true
      break
    case 'pause':
      playing = false
      pending = 0
      break
    case 'reset':
      playing = false
      pending = 0
      t.reset()
      lastTestRound = 0
      break
    case 'setEta':
      t.eta = cmd.eta
      break
    case 'setSpeed':
      speed = cmd.speed
      break
    case 'setTestEvery':
      testEvery = cmd.rounds
      break
    case 'setExamples':
      playing = false
      pending = 0
      t.setExamples(cmd.changes)
      lastTestRound = 0
      break
    case 'setWeight':
      t.setWeight(cmd.layer, cmd.from, cmd.to, cmd.value)
      if (!playing && pending === 0) testNow(t)
      break
    case 'experiment':
      playing = false
      t.eta = cmd.eta
      t.setExamples(cmd.changes)
      lastTestRound = 0
      pending = cmd.rounds
      experiment = { id: cmd.id, probe: cmd.probe }
      break
  }
  report(t)
  if (playing || pending > 0) void run(t)
}

self.onmessage = (event: MessageEvent<Command>) => {
  if (trainer) handle(trainer, event.data)
  else queue.push(event.data)
}

async function start() {
  post({ type: 'loading' })
  try {
    const mnist = await loadMnist()
    trainer = new Trainer(mnist)
    // The first example of each digit, for pages to show or replay.
    const firstOf = (data: typeof mnist.test) =>
      Array.from({ length: 10 }, (_, d) => Float32Array.from(getSample(data, data.labels.indexOf(d)).x))
    post({ type: 'ready', samples: firstOf(mnist.test), trainSamples: firstOf(mnist.train) })
    report(trainer)
    for (const cmd of queue.splice(0)) handle(trainer, cmd)
  } catch (error) {
    post({ type: 'error', message: error instanceof Error ? error.message : String(error) })
  }
}

void start()
