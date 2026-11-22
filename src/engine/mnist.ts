// MNIST data: parsing the IDX files, drawing batches, and scoring the network on a test set.
import {
  DIGITS,
  IMAGE_SIDE,
  PIXELS,
  createRng,
  forward,
  predict,
  type Network,
  type Sample,
} from './network'

/** Images stored back to back, pixel values scaled to [0, 1]. */
export interface Dataset {
  count: number
  images: Float32Array
  labels: Uint8Array
}

const IMAGES_MAGIC = 0x00000803
const LABELS_MAGIC = 0x00000801

/** Parses an IDX3 image file (e.g. train-images-idx3-ubyte). */
export function parseImages(buffer: ArrayBuffer): Float32Array {
  const view = new DataView(buffer)
  if (view.getUint32(0) !== IMAGES_MAGIC) throw new Error('Not an IDX image file')
  const count = view.getUint32(4)
  const rows = view.getUint32(8)
  const cols = view.getUint32(12)
  if (rows !== IMAGE_SIDE || cols !== IMAGE_SIDE) throw new Error(`Expected 28 × 28 images, got ${rows} × ${cols}`)
  const bytes = new Uint8Array(buffer, 16, count * PIXELS)
  const images = new Float32Array(bytes.length)
  for (let k = 0; k < bytes.length; k++) images[k] = bytes[k] / 255
  return images
}

/** Parses an IDX1 label file (e.g. train-labels-idx1-ubyte). */
export function parseLabels(buffer: ArrayBuffer): Uint8Array {
  const view = new DataView(buffer)
  if (view.getUint32(0) !== LABELS_MAGIC) throw new Error('Not an IDX label file')
  return new Uint8Array(buffer, 8, view.getUint32(4)).slice()
}

export function createDataset(images: Float32Array, labels: Uint8Array): Dataset {
  if (images.length !== labels.length * PIXELS) {
    throw new Error(`${labels.length} labels but ${images.length / PIXELS} images`)
  }
  return { count: labels.length, images, labels }
}

export const MNIST_FILES = {
  trainImages: 'train-images-idx3-ubyte.gz',
  trainLabels: 'train-labels-idx1-ubyte.gz',
  testImages: 't10k-images-idx3-ubyte.gz',
  testLabels: 't10k-labels-idx1-ubyte.gz',
} as const

/** Gunzips if the bytes are still gzipped (some servers decompress .gz on the way). */
export async function gunzipIfNeeded(bytes: ArrayBuffer): Promise<ArrayBuffer> {
  const head = new Uint8Array(bytes, 0, Math.min(2, bytes.byteLength))
  if (head[0] !== 0x1f || head[1] !== 0x8b) return bytes
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))
  return new Response(stream).arrayBuffer()
}

export interface Mnist {
  train: Dataset
  test: Dataset
}

/** Reads one file's raw bytes by name. */
export type FetchBytes = (name: string) => Promise<ArrayBuffer>

const fetchFromPublic: FetchBytes = async (name) => {
  const response = await fetch(`${import.meta.env.BASE_URL}mnist/${name}`)
  if (!response.ok) throw new Error(`Could not load ${name}: ${response.status}`)
  return response.arrayBuffer()
}

/** Loads the 60,000 training and 10,000 test digits (from public/mnist by default). */
export async function loadMnist(fetchBytes: FetchBytes = fetchFromPublic): Promise<Mnist> {
  const read = async (name: string) => gunzipIfNeeded(await fetchBytes(name))
  const [trainImages, trainLabels, testImages, testLabels] = await Promise.all([
    read(MNIST_FILES.trainImages),
    read(MNIST_FILES.trainLabels),
    read(MNIST_FILES.testImages),
    read(MNIST_FILES.testLabels),
  ])
  return {
    train: createDataset(parseImages(trainImages), parseLabels(trainLabels)),
    test: createDataset(parseImages(testImages), parseLabels(testLabels)),
  }
}

export function getSample(data: Dataset, index: number): Sample {
  return { x: data.images.subarray(index * PIXELS, (index + 1) * PIXELS), label: data.labels[index] }
}

/** How many examples of each digit (the playground's "Its whole experience" bars). */
export function labelCounts(data: Dataset): number[] {
  const counts = new Array<number>(DIGITS).fill(0)
  for (const label of data.labels) counts[label]++
  return counts
}

/** Keeps only the examples `keep` accepts, e.g. to remove every 7. */
export function filterDataset(data: Dataset, keep: (label: number, index: number) => boolean): Dataset {
  const indices: number[] = []
  for (let k = 0; k < data.count; k++) if (keep(data.labels[k], k)) indices.push(k)
  const images = new Float32Array(indices.length * PIXELS)
  const labels = new Uint8Array(indices.length)
  indices.forEach((src, dst) => {
    images.set(data.images.subarray(src * PIXELS, (src + 1) * PIXELS), dst * PIXELS)
    labels[dst] = data.labels[src]
  })
  return { count: indices.length, images, labels }
}

/** A shuffled order of the dataset, split into batches. One pass = one epoch. */
export function batchOrder(count: number, batchSize: number, seed: number): number[][] {
  const order = Array.from({ length: count }, (_, k) => k)
  const rand = createRng(seed)
  for (let k = count - 1; k > 0; k--) {
    const swap = Math.floor(rand() * (k + 1))
    ;[order[k], order[swap]] = [order[swap], order[k]]
  }
  const batches: number[][] = []
  for (let start = 0; start < count; start += batchSize) batches.push(order.slice(start, start + batchSize))
  return batches
}

export const toBatch = (data: Dataset, indices: readonly number[]): Sample[] =>
  indices.map((k) => getSample(data, k))

export interface Evaluation {
  /** Fraction of all examples read correctly. */
  accuracy: number
  /** Fraction read correctly for each true digit. */
  perDigit: number[]
  /** confusion[actual][predicted] counts. */
  confusion: number[][]
}

export function evaluate(net: Network, data: Dataset): Evaluation {
  const confusion = Array.from({ length: DIGITS }, () => new Array<number>(DIGITS).fill(0))
  let correct = 0
  for (let k = 0; k < data.count; k++) {
    const { x, label } = getSample(data, k)
    const guess = predict(forward(net, x).yHat)
    confusion[label][guess]++
    if (guess === label) correct++
  }
  const perDigit = confusion.map((row, d) => {
    const total = row.reduce((a, b) => a + b, 0)
    return total === 0 ? 0 : row[d] / total
  })
  return { accuracy: data.count === 0 ? 0 : correct / data.count, perDigit, confusion }
}
