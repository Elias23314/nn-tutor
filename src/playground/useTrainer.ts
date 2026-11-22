import { useCallback, useEffect, useRef, useState } from 'react'
import type { Command, Report, Snapshot } from './protocol'

export type TrainerStatus =
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'ready'; state: Snapshot; samples: Float32Array[]; trainSamples: Float32Array[] }

export type Send = (command: Command) => void

/**
 * Starts a training worker for this page and keeps the latest snapshot of its state.
 * `initial` commands are sent as soon as the worker starts (it queues them until its data loads).
 */
export function useTrainer(initial: Command[] = []) {
  const worker = useRef<Worker | null>(null)
  const samples = useRef<{ test: Float32Array[]; train: Float32Array[] } | null>(null)
  const [status, setStatus] = useState<TrainerStatus>({ kind: 'loading' })
  const initialCommands = useRef(initial)

  useEffect(() => {
    const w = new Worker(new URL('./trainer.worker.ts', import.meta.url), { type: 'module' })
    worker.current = w
    w.onmessage = (event: MessageEvent<Report>) => {
      const report = event.data
      if (report.type === 'ready') samples.current = { test: report.samples, train: report.trainSamples }
      else if (report.type === 'error') setStatus({ kind: 'error', message: report.message })
      else if (report.type === 'state' && samples.current) {
        const { test, train } = samples.current
        setStatus({ kind: 'ready', state: report.state, samples: test, trainSamples: train })
      }
    }
    w.onerror = (event) => setStatus({ kind: 'error', message: event.message || 'The training worker failed' })
    for (const cmd of initialCommands.current) w.postMessage(cmd)
    return () => {
      w.terminate()
      worker.current = null
    }
  }, [])

  const send = useCallback<Send>((command) => worker.current?.postMessage(command), [])
  return { status, send }
}
