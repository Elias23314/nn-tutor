// The MNIST network from the mockups: 784 pixels → 16 → 16 → 10.
// ReLU in both hidden layers, sigmoid outputs, loss = average of ten squared gaps.
// Names follow docs/notation.md.

export const IMAGE_SIDE = 28
export const PIXELS = IMAGE_SIDE * IMAGE_SIDE // 784
export const DIGITS = 10
export const LAYER_SIZES: readonly number[] = [PIXELS, 16, 16, DIGITS]

/** Learning rate η. The loss averages ten outputs, so gradients are small and η is large. */
export const DEFAULT_ETA = 2
/** Examples per round of practice. 60,000 / 32 = 1,875 rounds per pass. */
export const BATCH_SIZE = 32

/**
 * One layer of connections. `weights[j * inSize + i]` is the weight from neuron i
 * of the previous layer to neuron j of this layer.
 */
export interface Layer {
  inSize: number
  outSize: number
  weights: Float64Array
  biases: Float64Array
}

/** layers[0]: pixels → hidden 1, layers[1]: hidden 1 → hidden 2, layers[2]: hidden 2 → outputs. */
export interface Network {
  layers: Layer[]
}

/** ∂L/∂p for every parameter, shaped like the network. */
export type Gradients = Network

export interface LayerActivity {
  /** zⱼ = Σᵢ wᵢⱼ aᵢ + bⱼ */
  z: Float64Array
  /** aⱼ = ReLU(zⱼ) in hidden layers, ŷⱼ = σ(zⱼ) in the output layer. */
  a: Float64Array
}

export interface ForwardResult {
  /** Pixel values x, each in [0, 1]. */
  x: ArrayLike<number>
  /** One entry per layer of connections; the last one is the output layer. */
  layers: LayerActivity[]
  /** ŷ: the ten output activations. */
  yHat: Float64Array
}

export interface Sample {
  x: ArrayLike<number>
  /** The digit 0–9. Its goal vector y is one-hot. */
  label: number
}

// ---------- Naming ----------

/** Hidden layer 1 neurons are a–p, hidden layer 2 neurons are A–P. */
export function neuronName(hiddenLayer: 1 | 2, j: number): string {
  const letter = String.fromCharCode(97 + j)
  return hiddenLayer === 1 ? letter : letter.toUpperCase()
}

/** Pixel index for a 1-based row and column, as the mockups count them. */
export const pixelIndex = (row: number, column: number): number =>
  (row - 1) * IMAGE_SIDE + (column - 1)

export function weight(net: Network, layer: number, from: number, to: number): number {
  const l = net.layers[layer]
  return l.weights[to * l.inSize + from]
}

export const countParams = (net: Network): number =>
  net.layers.reduce((n, l) => n + l.weights.length + l.biases.length, 0)

// ---------- Setup ----------

/** Small seeded PRNG (mulberry32) so every learner starts from the same network. */
export function createRng(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export const DEFAULT_SEED = 7

/**
 * Uniform random weights, zero biases. ReLU layers use He scaling, √(6 / in).
 * The sigmoid output layer starts small, ¼ · √(6 / (in + out)). With larger output weights,
 * an output whose weights happen to be mostly negative is pushed far below 0 as the hidden
 * layers grow more active in the first rounds. The sigmoid then saturates (ŷ(1 − ŷ) ≈ 0.001)
 * and that digit is never learned: on MNIST, this happened for the 8 output with several seeds.
 */
export function createNetwork(sizes: readonly number[] = LAYER_SIZES, seed = DEFAULT_SEED): Network {
  const rand = createRng(seed)
  const layers: Layer[] = []
  for (let l = 1; l < sizes.length; l++) {
    const inSize = sizes[l - 1]
    const outSize = sizes[l]
    const isOutput = l === sizes.length - 1
    const limit = isOutput ? 0.25 * Math.sqrt(6 / (inSize + outSize)) : Math.sqrt(6 / inSize)
    const weights = new Float64Array(inSize * outSize)
    for (let k = 0; k < weights.length; k++) weights[k] = (rand() * 2 - 1) * limit
    layers.push({ inSize, outSize, weights, biases: new Float64Array(outSize) })
  }
  return { layers }
}

export function cloneNetwork(net: Network): Network {
  return {
    layers: net.layers.map((l) => ({
      inSize: l.inSize,
      outSize: l.outSize,
      weights: l.weights.slice(),
      biases: l.biases.slice(),
    })),
  }
}

export function zeroGradients(net: Network): Gradients {
  return {
    layers: net.layers.map((l) => ({
      inSize: l.inSize,
      outSize: l.outSize,
      weights: new Float64Array(l.weights.length),
      biases: new Float64Array(l.biases.length),
    })),
  }
}

// ---------- Activations and loss ----------

export const relu = (z: number): number => (z > 0 ? z : 0)
/** ReLU'(z), written r in the mockups. Taken as 0 at z = 0. */
export const reluDerivative = (z: number): number => (z > 0 ? 1 : 0)
export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z))

export function oneHot(label: number, size = DIGITS): Float64Array {
  const y = new Float64Array(size)
  y[label] = 1
  return y
}

/** L = (1/10) Σₖ (ŷₖ − yₖ)² */
export function loss(yHat: ArrayLike<number>, label: number): number {
  let sum = 0
  for (let k = 0; k < yHat.length; k++) sum += (yHat[k] - (k === label ? 1 : 0)) ** 2
  return sum / yHat.length
}

/** ∂L/∂ŷₖ = (2/10)(ŷₖ − yₖ) */
export const lossDerivative = (yHatK: number, yK: number, outputs = DIGITS): number =>
  (2 / outputs) * (yHatK - yK)

/** The digit with the highest output. */
export function predict(yHat: ArrayLike<number>): number {
  let best = 0
  for (let k = 1; k < yHat.length; k++) if (yHat[k] > yHat[best]) best = k
  return best
}

// ---------- Forward and backward ----------

/**
 * Indices of the nonzero inputs. Most MNIST pixels are blank (0), and a blank pixel adds
 * nothing to any weighted sum or gradient, so the first layer only visits these.
 */
function nonzero(x: ArrayLike<number>): Int32Array {
  const idx = new Int32Array(x.length)
  let n = 0
  for (let i = 0; i < x.length; i++) if (x[i] !== 0) idx[n++] = i
  return idx.subarray(0, n)
}

export function forward(net: Network, x: ArrayLike<number>): ForwardResult {
  const layers: LayerActivity[] = []
  const last = net.layers.length - 1
  const active = nonzero(x)
  let input: Float64Array | null = null
  net.layers.forEach((layer, l) => {
    const { inSize, outSize, weights, biases } = layer
    const z = new Float64Array(outSize)
    const a = new Float64Array(outSize)
    for (let j = 0; j < outSize; j++) {
      let sum = biases[j]
      const row = j * inSize
      if (input === null) {
        for (let k = 0; k < active.length; k++) sum += weights[row + active[k]] * x[active[k]]
      } else {
        for (let i = 0; i < inSize; i++) sum += weights[row + i] * input[i]
      }
      z[j] = sum
      a[j] = l === last ? sigmoid(sum) : relu(sum)
    }
    layers.push({ z, a })
    input = a
  })
  return { x, layers, yHat: layers[last].a }
}

/**
 * Backpropagation for one sample. Adds ∂L/∂p into `into` (scaled by `scale`) and returns it,
 * so a batch can accumulate without allocating.
 */
export function backward(
  net: Network,
  fwd: ForwardResult,
  label: number,
  into: Gradients = zeroGradients(net),
  scale = 1,
): Gradients {
  const last = net.layers.length - 1
  const outputs = net.layers[last].outSize

  // δ = ∂L/∂z for the current layer. Output: ∂L/∂ŷₖ · ŷₖ(1 − ŷₖ).
  let delta = new Float64Array(outputs)
  for (let k = 0; k < outputs; k++) {
    const yHat = fwd.yHat[k]
    delta[k] = lossDerivative(yHat, k === label ? 1 : 0, outputs) * yHat * (1 - yHat)
  }

  for (let l = last; l >= 0; l--) {
    const { inSize, outSize, weights } = net.layers[l]
    const grad = into.layers[l]
    const active = l === 0 ? nonzero(fwd.x) : null

    for (let j = 0; j < outSize; j++) {
      const d = delta[j] * scale
      if (d === 0) continue // silent neuron: every connection into it gets 0
      grad.biases[j] += d
      const row = j * inSize
      if (active) {
        // Blank pixels get 0, so only the inked ones are visited.
        const x = fwd.x
        for (let k = 0; k < active.length; k++) grad.weights[row + active[k]] += d * x[active[k]]
      } else {
        const input = fwd.layers[l - 1].a
        for (let i = 0; i < inSize; i++) grad.weights[row + i] += d * input[i]
      }
    }

    if (l > 0) {
      // ∂L/∂zᵢ (previous layer) = r(zᵢ) · Σⱼ wᵢⱼ δⱼ
      const prevZ = fwd.layers[l - 1].z
      const prev = new Float64Array(inSize)
      for (let i = 0; i < inSize; i++) {
        if (reluDerivative(prevZ[i]) === 0) continue
        let sum = 0
        for (let j = 0; j < outSize; j++) sum += weights[j * inSize + i] * delta[j]
        prev[i] = sum
      }
      delta = prev
    }
  }
  return into
}

/** One gradient-descent update: p ← p − η · ∂L/∂p. Returns a new network. */
export function applyGradients(net: Network, grads: Gradients, eta = DEFAULT_ETA): Network {
  const next = cloneNetwork(net)
  next.layers.forEach((layer, l) => {
    const g = grads.layers[l]
    for (let k = 0; k < layer.weights.length; k++) layer.weights[k] -= eta * g.weights[k]
    for (let k = 0; k < layer.biases.length; k++) layer.biases[k] -= eta * g.biases[k]
  })
  return next
}

export interface StepResult {
  /** Average loss over the batch, before the update. */
  loss: number
  /** Average gradient over the batch. */
  grads: Gradients
  network: Network
}

/** One round of practice: forward, backward and an update on the averaged gradient of a batch. */
export function trainStep(net: Network, batch: readonly Sample[], eta = DEFAULT_ETA): StepResult {
  if (batch.length === 0) throw new Error('trainStep needs at least one sample')
  const grads = zeroGradients(net)
  const scale = 1 / batch.length
  let total = 0
  for (const sample of batch) {
    const fwd = forward(net, sample.x)
    total += loss(fwd.yHat, sample.label)
    backward(net, fwd, sample.label, grads, scale)
  }
  return { loss: total / batch.length, grads, network: applyGradients(net, grads, eta) }
}
